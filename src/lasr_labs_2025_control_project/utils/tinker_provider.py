"""Single-file Tinker (Thinking Machines) compatibility layer for the pipeline.

Everything Tinker-specific lives in this module. It does two jobs:

1. Registers an inspect-ai model provider under the ``tinker/`` prefix, so that
   ``get_model("tinker/Qwen/Qwen3-32B")`` and ``eval(model="tinker/...")`` work
   for solution *generation* (attacks / honest / honeypots), the text-``<score>``
   suspiciousness monitors, and the LLM paraphraser -- i.e. every plain-sampling
   role -- with no other pipeline changes.

2. Exposes :func:`collusion_yes_no_logprobs`, which returns the *exact*
   log P(YES) / log P(NO) answer-token logprobs the untrusted collusion monitor
   needs, via forced decoding (``compute_logprobs`` on ``prompt+"YES"`` vs
   ``prompt+"NO"``). This sidesteps the top-k truncation / provider-shape / CoT
   contamination problems that plague logprobs on OpenAI-compatible backends,
   and is the correct method for a binary-decision monitor.

The ``tinker`` package is imported lazily (only when a tinker model is actually
used) so the rest of the pipeline runs without it installed. Auth is via the
``TINKER_API_KEY`` environment variable, read by the Tinker SDK itself.

Model ids are the part after the ``tinker/`` prefix, e.g. ``Qwen/Qwen3-32B``.
The authoritative live catalogue is
``ServiceClient().get_server_capabilities().supported_models``.
"""

from __future__ import annotations

import asyncio
import threading
from functools import cache
from typing import TYPE_CHECKING, Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.model import (
    modelapi as _modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

if TYPE_CHECKING:  # avoid importing tinker at module load
    from tinker import SamplingClient, ServiceClient

TINKER_PREFIX = "tinker/"

# Default sampling cap when the caller doesn't set one. This governs every
# free-generating call that passes no max_tokens: the <score> suspiciousness
# monitors and the LLM paraphraser. (Generator configs set their own budget.)
#
# It must be generous, because truncation here is *silent*: a cut-off monitor
# completion has no closing </score>, so parse_score returns None -> NaN, which
# is indistinguishable from a refusal; and a cut-off paraphrase is stored
# verbatim as the transformed solution and fed to every downstream monitor.
#
# Measured on the APPS open run at the previous value of 4096: 24 repeat calls
# to T (Qwen3.5-4B) on one monitor prompt gave 6 parse failures, *all six* with
# stop_reason=max_tokens and none of the 18 successes truncated -- i.e. 100% of
# NaN scores were truncation, up to 11% of samples on an arm. The closed-model
# run has 0% NaN on the same monitors, so this is an open-model artefact: small
# chatty models emit far more reasoning before the tag than GPT-4o-mini does.
_DEFAULT_MAX_TOKENS = 16384

# Chat-template kwargs that disable chain-of-thought for hybrid-reasoning
# families, so completions carry only the final answer (matches the behaviour
# we rely on for clean answer extraction). Applied best-effort: templates that
# don't accept the kwarg are called without it.
# - enable_thinking=False: Qwen / DeepSeek
# - thinking=False: Kimi K2.5 / K2.6
_NOTHINK_TEMPLATE_KWARGS = {"enable_thinking": False, "thinking": False}

# inspect's reasoning_effort vocabulary -> the values harmony templates (gpt-oss)
# accept. inspect allows "minimal", harmony does not; "low" is the nearest.
_HARMONY_REASONING_EFFORT = {"minimal": "low"}


def _ensure_tiktoken_blobfile_patch() -> None:
    """tiktoken.load.read_file requires blobfile for local paths.

    If blobfile is not installed, fallback to standard Python open(..., 'rb').
    This enables Kimi K2.6 (and other tiktoken-based tokenizers) to load
    cleanly without failing on missing blobfile.
    """
    try:
        import tiktoken.load

        if getattr(tiktoken.load, "_spar_patched", False):
            return

        def _read_file_fallback(blobpath: str) -> bytes:
            if blobpath.startswith("http://") or blobpath.startswith("https://"):
                import requests

                resp = requests.get(blobpath)
                resp.raise_for_status()
                return resp.content
            try:
                import blobfile

                with blobfile.BlobFile(blobpath, "rb") as f:
                    return f.read()
            except ImportError:
                with open(blobpath, "rb") as f:
                    return f.read()

        tiktoken.load.read_file = _read_file_fallback
        tiktoken.load._spar_patched = True
    except (ImportError, AttributeError):
        pass


_ensure_tiktoken_blobfile_patch()


def _load_fast_tokenizer_fallback(base_model: str):
    """Fallback tokenizer loader for models with non-standard tokenizer_class.

    For example, zai-org/GLM-5.3 has "tokenizer_class": "TokenizersBackend" in
    tokenizer_config.json, which causes AutoTokenizer.from_pretrained() to fail
    with "ValueError: Tokenizer class TokenizersBackend does not exist".
    This loads tokenizer.json and chat_template.jinja directly with PreTrainedTokenizerFast.
    """
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    repo_id = base_model.split(":")[0]
    tok_file = hf_hub_download(repo_id, "tokenizer.json")
    kwargs: dict[str, Any] = {}
    if "GLM" in repo_id or "glm" in repo_id:
        kwargs["eos_token"] = "<|endoftext|>"
        kwargs["pad_token"] = "<|endoftext|>"
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file, **kwargs)

    try:
        chat_template_file = hf_hub_download(repo_id, "chat_template.jinja")
        with open(chat_template_file, encoding="utf-8") as f:
            tok.chat_template = f.read()
    except Exception:
        pass
    return tok


def strip_prefix(model_name: str) -> str:
    """Return the Tinker base-model id from a possibly-``tinker/``-prefixed name."""
    return model_name.removeprefix(TINKER_PREFIX)


def is_tinker_model(model_name: str) -> bool:
    return model_name.startswith(TINKER_PREFIX)


# --------------------------------------------------------------------------- #
# Client / tokenizer caches (creating a SamplingClient is near-instant, but a
# process-wide cache keeps us from rebuilding one per request).
# --------------------------------------------------------------------------- #
_service_lock = threading.Lock()
_service_client: ServiceClient | None = None


def _service() -> ServiceClient:
    global _service_client
    if _service_client is None:
        with _service_lock:
            if _service_client is None:
                import os

                if not os.environ.get("TINKER_API_KEY"):
                    raise RuntimeError(
                        "TINKER_API_KEY is not set; required to use tinker/ models."
                    )
                import tinker

                _service_client = tinker.ServiceClient()
    return _service_client


@cache
def _sampling_client(model_ref: str) -> SamplingClient:
    # A `tinker://…` ref is a saved (e.g. LoRA-finetuned) checkpoint, loaded via
    # model_path; anything else is a base model name.
    if model_ref.startswith("tinker://"):
        return _service().create_sampling_client(model_path=model_ref)
    return _service().create_sampling_client(base_model=model_ref)


@cache
def _tokenizer(model_ref: str):
    _ensure_tiktoken_blobfile_patch()
    sc = _sampling_client(model_ref)
    try:
        return sc.get_tokenizer()
    except ValueError as e:
        if "TokenizersBackend" in str(
            e
        ) or "does not exist or is not currently imported" in str(e):
            base_model = (
                sc.get_base_model()
                if hasattr(sc, "get_base_model")
                else strip_prefix(model_ref)
            )
            return _load_fast_tokenizer_fallback(base_model)
        raise


def _normalize_ids(out: Any) -> list[int]:
    """Coerce apply_chat_template output to a flat list[int].

    transformers >=5 returns a BatchEncoding (``{"input_ids": [...]}``) rather
    than a bare list; some versions/paths also nest a batch dimension.
    """
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def _apply_chat_template(
    base_model: str,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    reasoning_effort: str | None = None,
) -> list[int]:
    """Tokenise chat messages, minimising thinking when the template supports it.

    Two independent, best-effort levers (templates that ignore a kwarg simply
    render as usual):
      - `enable_thinking=False` for hybrid-reasoning families (Qwen/DeepSeek).
      - `reasoning_effort` for harmony-style templates (gpt-oss), which default
        to `Reasoning: medium` and ignore `enable_thinking` entirely. Tinker's
        *sampling* API has no such knob, but the chat template does, and we are
        the caller of the template.
    """
    tok = _tokenizer(base_model)
    kwargs = dict(_NOTHINK_TEMPLATE_KWARGS)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = _HARMONY_REASONING_EFFORT.get(
            reasoning_effort, reasoning_effort
        )
    try:
        out = tok.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **kwargs,
        )
    except TypeError:
        out = tok.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
        )
    return _normalize_ids(out)


def _messages_to_dicts(input: list[ChatMessage]) -> list[dict[str, str]]:
    role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }
    return [{"role": role_map.get(m.role, "user"), "content": m.text} for m in input]


# --------------------------------------------------------------------------- #
# inspect-ai provider: handles plain sampling (generation / text monitors /
# paraphrasing). Collusion logprobs go through collusion_yes_no_logprobs below.
# --------------------------------------------------------------------------- #
@_modelapi(name="tinker")
def tinker() -> type[ModelAPI]:
    return TinkerAPI


class TinkerAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["TINKER_API_KEY"],
            config=config,
        )
        # model_name arrives with the tinker/ prefix already stripped by get_model
        self.base_model = strip_prefix(model_name)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        from tinker import types

        prompt_ids = _apply_chat_template(
            self.base_model,
            _messages_to_dicts(input),
            add_generation_prompt=True,
            reasoning_effort=config.reasoning_effort,
        )
        sampling_params = types.SamplingParams(
            # Generous default for callers that set no max_tokens (monitors, the
            # paraphraser): a short cap truncates generators mid-output, e.g.
            # before the closing </final_code> tag. Generator configs override
            # this via RunGenerator.max_tokens.
            max_tokens=config.max_tokens or _DEFAULT_MAX_TOKENS,
            temperature=1.0 if config.temperature is None else config.temperature,
            top_p=1.0 if config.top_p is None else config.top_p,
            top_k=-1 if config.top_k is None else config.top_k,
            stop=config.stop_seqs,
            seed=config.seed,
        )
        sc = _sampling_client(self.base_model)
        resp = await sc.sample_async(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=sampling_params,
        )
        seq = resp.sequences[0]
        text = _tokenizer(self.base_model).decode(
            list(seq.tokens), skip_special_tokens=True
        )
        # tinker's StopReason is Literal["length", "stop"] (lowercase); compare
        # case-insensitively or truncation is silently reported as a clean stop,
        # which is exactly how the generator truncation went unnoticed in a pilot.
        stop_reason = (
            "max_tokens" if str(seq.stop_reason).lower().endswith("length") else "stop"
        )
        return ModelOutput.from_content(
            model=self.model_name, content=text, stop_reason=stop_reason
        )

    def connection_key(self) -> str:
        return f"tinker/{self.base_model}"


# --------------------------------------------------------------------------- #
# Forced-decoding YES/NO logprobs for the untrusted collusion monitor.
# --------------------------------------------------------------------------- #
async def collusion_yes_no_logprobs(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    answer_open: str = "<answer>",
    yes_text: str = "YES",
    no_text: str = "NO",
) -> dict[str, Any]:
    """Return exact answer-token logprobs for a YES/NO collusion monitor.

    Builds the assistant scaffold up to ``<answer>`` and teacher-forces both
    ``YES`` and ``NO`` continuations, reading each one's summed token logprob
    (handles multi-token YES/NO). Returns ``{"YES", "NO", "completion"}`` in the
    same shape as ``get_answer_logprobs_from_openai_completion`` so the collusion
    scorer treats it identically.
    """
    from tinker import types

    base_model = strip_prefix(model_name)
    tok = _tokenizer(base_model)
    sc = _sampling_client(base_model)

    prefix_ids = list(
        _apply_chat_template(
            base_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
        )
    )
    # If the template leaves an unclosed <think> tag (e.g. GLM-5.3 template unconditionally
    # appends <|assistant|><think>), close it before appending <answer> so forced decoding
    # evaluates the actual answer rather than reasoning tokens.
    if prefix_ids and hasattr(tok, "convert_tokens_to_ids"):
        think_id = tok.convert_tokens_to_ids("<think>")
        end_think_id = tok.convert_tokens_to_ids("</think>")
        if (
            isinstance(think_id, int)
            and isinstance(end_think_id, int)
            and prefix_ids[-1] == think_id
        ):
            prefix_ids.append(end_think_id)

    prefix_ids += tok.encode(answer_open, add_special_tokens=False)

    async def _candidate_logprob(text: str) -> float | None:
        cand_ids = tok.encode(text, add_special_tokens=False)
        if not cand_ids:
            return None
        full = prefix_ids + cand_ids
        lps = await sc.compute_logprobs_async(types.ModelInput.from_ints(full))
        # compute_logprobs returns one logprob per input token (None/NaN where
        # undefined). The candidate occupies the final len(cand_ids) positions.
        tail = lps[len(prefix_ids) :]
        vals = [lp for lp in tail if lp is not None and lp == lp]  # drop None/NaN
        if len(vals) != len(cand_ids):
            return None
        return float(sum(vals))

    yes_lp, no_lp = await asyncio.gather(
        _candidate_logprob(yes_text),
        _candidate_logprob(no_text),
    )

    if yes_lp is not None and no_lp is not None:
        answer = yes_text if yes_lp >= no_lp else no_text
    elif yes_lp is not None:
        answer = yes_text
    elif no_lp is not None:
        answer = no_text
    else:
        answer = ""
    completion = f"{answer_open}{answer}</answer>" if answer else ""

    return {"YES": yes_lp, "NO": no_lp, "completion": completion}
