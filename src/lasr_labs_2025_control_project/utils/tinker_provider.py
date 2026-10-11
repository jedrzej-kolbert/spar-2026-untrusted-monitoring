"""Single-file Tinker (Thinking Machines) compatibility layer for the pipeline.

Everything Tinker-specific lives in this module. It does two jobs:

1. Registers an inspect-ai model provider under the ``tinker/`` prefix, so that
   ``get_model("tinker/Qwen/Qwen3-32B")`` and ``eval(model="tinker/...")`` work
   for solution *generation* (attacks / honest / honeypots), the text-``<score>``
   suspiciousness monitors, and the LLM paraphraser -- i.e. every plain-sampling
   role -- with no other pipeline changes.

2. Exposes :func:`collusion_yes_no_logprobs`, which returns the *exact*
   log P(YES) / log P(NO) answer-token logprobs the untrusted collusion monitor
   needs, via full teacher-forced continuations by default, or opt-in targeted
   prompt logprobs in one prefill for single-token answers. This sidesteps top-k / CoT
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

import inspect
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
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
_NOTHINK_TEMPLATE_KWARGS = {"enable_thinking": False}

# inspect's reasoning_effort vocabulary -> the values harmony templates (gpt-oss)
# accept. inspect allows "minimal", harmony does not; "low" is the nearest.
_HARMONY_REASONING_EFFORT = {"minimal": "low"}


def strip_prefix(model_name: str) -> str:
    """Return the Tinker base-model id from a possibly-``tinker/``-prefixed name."""
    return (
        model_name[len(TINKER_PREFIX) :]
        if model_name.startswith(TINKER_PREFIX)
        else model_name
    )


def is_tinker_model(model_name: str) -> bool:
    return model_name.startswith(TINKER_PREFIX)


# --------------------------------------------------------------------------- #
# Client / tokenizer caches (creating a SamplingClient is near-instant, but a
# process-wide cache keeps us from rebuilding one per request).
# --------------------------------------------------------------------------- #
_service_lock = threading.Lock()
_service_client: "Optional[ServiceClient]" = None


def _service() -> "ServiceClient":
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


@lru_cache(maxsize=None)
def _sampling_client(model_ref: str) -> "SamplingClient":
    # A `tinker://…` ref is a saved (e.g. LoRA-finetuned) checkpoint, loaded via
    # model_path; anything else is a base model name.
    if model_ref.startswith("tinker://"):
        return _service().create_sampling_client(model_path=model_ref)
    return _service().create_sampling_client(base_model=model_ref)


@lru_cache(maxsize=None)
def _tokenizer(model_ref: str):
    return _sampling_client(model_ref).get_tokenizer()


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
        output = ModelOutput.from_content(
            model=self.model_name, content=text, stop_reason=stop_reason
        )
        # Token counts so eval logs record usage (cost estimates); Tinker bills the
        # full prompt on every call.
        output.usage = ModelUsage(
            input_tokens=len(prompt_ids),
            output_tokens=len(seq.tokens),
            total_tokens=len(prompt_ids) + len(seq.tokens),
        )
        return output

    def connection_key(self) -> str:
        return f"tinker/{self.base_model}"


# --------------------------------------------------------------------------- #
# Forced-decoding YES/NO logprobs for the untrusted collusion monitor.
# --------------------------------------------------------------------------- #
def _answer_target_logprobs(tensor: Any, prefix_length: int) -> list[Optional[float]]:
    """Read only requested cells; an absent CSR entry is not a zero logprob."""
    if tensor is None:
        return [None, None]
    if list(tensor.shape or []) != [prefix_length, 2]:
        raise ValueError("Unexpected YES/NO target logprob shape")
    values = tensor.data
    crow = tensor.sparse_crow_indices
    col = tensor.sparse_col_indices
    if crow is None:
        if col is not None or len(values) != prefix_length * 2:
            raise ValueError("Malformed dense YES/NO target logprobs")
        selected = values[-2:]
    else:
        if (
            col is None
            or len(crow) != prefix_length + 1
            or crow[0] != 0
            or crow[-1] != len(values)
            or len(col) != len(values)
            or any(a > b for a, b in zip(crow, crow[1:]))
            or any(c not in (0, 1) for c in col)
        ):
            raise ValueError("Malformed sparse YES/NO target logprobs")
        start, end = crow[-2:]
        requested = col[start:end]
        if len(set(requested)) != len(requested):
            raise ValueError("Duplicate YES/NO target logprobs")
        row = dict(zip(requested, values[start:end]))
        selected = [row.get(0), row.get(1)]
    # Preserve the teacher-forced path's None/NaN and infinity semantics.
    return [float(v) if v is not None and v == v else None for v in selected]


async def collusion_yes_no_logprobs(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    answer_open: str = "<answer>",
    yes_text: str = "YES",
    no_text: str = "NO",
    single_prefill: bool = False,
) -> dict[str, Any]:
    """Return exact answer-token logprobs for a YES/NO collusion monitor.

    Builds the assistant scaffold up to ``<answer>`` and scores both continuations
    exactly. Experimental ``single_prefill=True`` lets single-token answers share
    one prefill on SDKs with targeted prompt logprobs. Full teacher forcing remains
    the default and the fallback for multi-token answers and older SDKs.
    Returns ``{"YES", "NO", "completion"}`` in the
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
    prefix_ids += tok.encode(answer_open, add_special_tokens=False)

    yes_ids = tok.encode(yes_text, add_special_tokens=False)
    no_ids = tok.encode(no_text, add_special_tokens=False)

    async def _candidate_logprob(cand_ids: list[int]) -> Optional[float]:
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

    tensor_type = getattr(types, "TensorData", None)
    if (
        single_prefill
        and prefix_ids
        and len(yes_ids) == len(no_ids) == 1
        and tensor_type is not None
        and hasattr(tensor_type, "from_torch_sparse")
        and "target_prompt_logprobs" in inspect.signature(sc.sample_async).parameters
    ):
        import torch

        # Row i scores prompt position i+1. Append an ordinary placeholder so
        # row P-1 conditions on all P original prefix tokens, never the placeholder.
        targets = torch.full((len(prefix_ids), 2), -1, dtype=torch.int64)
        targets[-1] = torch.tensor([yes_ids[0], no_ids[0]], dtype=torch.int64)
        resp = await sc.sample_async(
            prompt=types.ModelInput.from_ints(prefix_ids + yes_ids),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            # Keep the legacy prompt-logprob computation enabled: target-only
            # mode gave materially different scores on a live DeepSeek checkpoint.
            include_prompt_logprobs=True,
            target_prompt_logprobs=tensor_type.from_torch_sparse(targets, pad_value=-1),
        )
        # The generated token after the placeholder is irrelevant. Do not retry
        # through teacher forcing on backend errors: auth/quota/context failures
        # must propagate rather than trigger additional paid requests.
        yes_lp, no_lp = _answer_target_logprobs(
            resp.target_prompt_logprobs, len(prefix_ids)
        )
    else:
        yes_lp = await _candidate_logprob(yes_ids)
        no_lp = await _candidate_logprob(no_ids)

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
