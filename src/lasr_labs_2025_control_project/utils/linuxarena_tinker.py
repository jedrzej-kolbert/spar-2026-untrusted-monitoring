"""Control Tower's Tinker bridge; optional dependencies are loaded only at launch."""

import copy
import dataclasses
import json
import os
import shlex
from importlib import import_module
from importlib.metadata import entry_points, version
from pathlib import Path
from typing import Any
from uuid import uuid4


class ToolDialect:
    """Translate emitted calls, retaining each original for the model's history."""

    def __init__(self, mode: str = "auto") -> None:
        if mode not in {"auto", "schema", "argv"}:
            raise ValueError(
                f"Unsupported tool dialect {mode!r}; use auto, schema or argv"
            )
        self.mode = mode
        self.originals: dict[str, tuple[str, dict[str, Any]]] = {}

    def translate(self, call: Any, names: set[str]) -> dict[str, Any] | None:
        original = (call.function, copy.deepcopy(call.arguments))
        if (
            self.mode != "schema"
            and call.function not in names
            and "." in call.function
        ):
            base = call.function.split(".", 1)[0]
            if base not in names:
                raise ValueError(
                    f"Unsupported tool name {call.function!r}; declared: {sorted(names)}"
                )
            call.function = base
        if call.function == "bash":
            for key in ("cmd", "command"):
                value = call.arguments.get(key)
                if isinstance(value, list):
                    if (
                        self.mode == "schema"
                        or not value
                        or not all(isinstance(a, str) for a in value)
                    ):
                        raise ValueError(
                            f"Unsupported bash {key} dialect: expected a string or nonempty string argv"
                        )
                    if (
                        len(value) == 3
                        and value[0] in {"bash", "sh", "/bin/bash", "/bin/sh"}
                        and value[1] in {"-c", "-lc"}
                    ):
                        call.arguments[key] = value[2]
                    else:
                        call.arguments[key] = shlex.join(value)
        if original != (call.function, call.arguments):
            self.originals[call.id] = original
            return {
                "call_id": call.id,
                "original_function": original[0],
                "original_arguments": original[1],
                "function": call.function,
                "arguments": call.arguments,
            }
        return None

    def native_history(self, messages: list[Any]) -> list[Any]:
        shown = []
        for message in messages:
            calls = getattr(message, "tool_calls", None)
            if message.role == "assistant" and calls:
                restored = []
                for call in calls:
                    original = self.originals.get(call.id)
                    if original is not None:
                        call = dataclasses.replace(
                            call,
                            function=original[0],
                            arguments=copy.deepcopy(original[1]),
                        )
                    elif (
                        self.mode == "argv"
                        and call.function == "bash"
                        and isinstance(call.arguments.get("cmd"), str)
                    ):
                        call = dataclasses.replace(
                            call,
                            arguments={
                                **call.arguments,
                                "cmd": ["bash", "-lc", call.arguments["cmd"]],
                            },
                        )
                    restored.append(call)
                message = message.model_copy(update={"tool_calls": restored})
            shown.append(message)
        return shown


def resolve_renderer(
    model: str, settings: dict[str, Any], monitoring: bool, model_info: Any
) -> str:
    """Use cookbook metadata, not model-name substring guesses."""
    key = "monitor_renderer" if monitoring else "renderer"
    explicit = settings.get(key)
    try:
        recommended = model_info.get_recommended_renderer_names(model)
    except Exception as exc:
        if explicit:
            return explicit
        raise ValueError(
            f"Unsupported model {model!r}: cookbook has no renderer metadata. Set an explicit {key} in LINUXARENA['model_settings'] for its role."
        ) from exc
    if explicit:
        if monitoring:
            disabled = [name for name in recommended if "disable_thinking" in name]
            if disabled and explicit not in disabled:
                raise ValueError(
                    f"Monitor renderer for {model!r} must disable thinking: {disabled}"
                )
        return explicit
    if monitoring:
        disabled = [name for name in recommended if "disable_thinking" in name]
        if disabled:
            return disabled[0]
    if not recommended:
        raise ValueError(f"Unsupported model {model!r}: empty renderer recommendations")
    return recommended[0]


def bounded_compose(data: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """Reject host access and bound every service, including sidecars."""
    data = copy.deepcopy(data)
    if data.get("include"):
        raise ValueError("Refusing unchecked external compose includes")
    for name, service in data.get("services", {}).items():
        if service.get("extends"):
            raise ValueError(f"Refusing unchecked inherited service {name!r}")
        if service.get("privileged") or any(
            service.get(k) == "host" for k in ("network_mode", "pid", "ipc")
        ):
            raise ValueError(f"Refusing privileged/host-namespace service {name!r}")
        if service.get("devices") or service.get("volumes_from"):
            raise ValueError(f"Refusing host devices or inherited mounts in {name!r}")
        for volume in service.get("volumes", []):
            if isinstance(volume, dict):
                source = str(volume.get("source", ""))
                unsafe = volume.get("type") == "bind" or source.startswith(
                    ("/", ".", "~")
                )
            else:
                source = str(volume).split(":", 1)[0]
                unsafe = ":" in str(volume) and (
                    source.startswith(("/", ".", "~")) or "/" in source or "$" in source
                )
            if unsafe:
                raise ValueError(f"Refusing host-mounted service {name!r}: {volume!r}")
        service.pop("build", None)
        if not service.get("image"):
            raise ValueError(
                f"No prebuilt image for {name!r}; local image builds are disabled"
            )
        prefix = "agent" if name == "default" else "sidecar"
        service["cpus"] = limits[f"{prefix}_cpus"]
        service["mem_limit"] = limits[f"{prefix}_memory"]
        service["memswap_limit"] = service["mem_limit"]
        service["pids_limit"] = limits["pids_limit"]
        service.setdefault("deploy", {}).setdefault("resources", {}).setdefault(
            "limits", {}
        ).update(cpus=service["cpus"], memory=service["mem_limit"])
    for name, volume in (data.get("volumes") or {}).items():
        if volume and (volume.get("driver_opts") or volume.get("external")):
            raise ValueError(
                f"Refusing host-backed or unchecked external named volume {name!r}"
            )
    return data


def register_provider(options: dict[str, Any]) -> None:
    """Register only inside the separate, pinned Control Tower interpreter."""
    if version("inspect-ai") != "0.3.268":
        raise RuntimeError(
            "LinuxArena requires Control Tower's Python with inspect-ai==0.3.268, not the project's environment"
        )
    cookbook = import_module("tinker_cookbook.eval.inspect_utils")
    model_info = import_module("tinker_cookbook.model_info")
    tinker = import_module("tinker")
    registry = import_module("inspect_ai.model._registry")
    tool_function = import_module("inspect_ai.tool").ToolFunction
    generate_config = import_module("inspect_ai.model").GenerateConfig
    clients: list[Any] = []

    for monitoring, route in (
        (False, "tinker-linuxarena"),
        (True, "tinker-linuxarena-monitor"),
    ):

        def make_api(is_monitor: bool) -> Any:
            class LinuxArenaTinkerAPI(cookbook.InspectAPIFromTinkerSampling):
                def __init__(
                    self, model_name: str, config: Any = None, **kwargs: Any
                ) -> None:
                    settings = options["models"][str(is_monitor)][model_name]
                    renderer = resolve_renderer(
                        model_name, settings, is_monitor, model_info
                    )
                    self.dialect = ToolDialect(settings["tool_dialect"])
                    if not clients:
                        clients.append(tinker.ServiceClient())
                    super().__init__(
                        renderer_name=renderer,
                        model_name=model_name,
                        sampling_client=clients[0].create_sampling_client(
                            base_model=model_name
                        ),
                        config=config or generate_config(),
                        **kwargs,
                    )

                async def generate(
                    self,
                    input: list[Any],
                    tools: list[Any],
                    tool_choice: Any,
                    config: Any,
                ) -> Any:
                    shown = self.dialect.native_history(input)
                    forced = tool_choice == "any" or isinstance(
                        tool_choice, tool_function
                    )
                    if is_monitor and forced:
                        if isinstance(tool_choice, tool_function):
                            requested = getattr(tool_choice, "name")
                            tools = [tool for tool in tools if tool.name == requested]
                        tool_choice = "auto"
                    output = await super().generate(shown, tools, tool_choice, config)
                    for choice in output.choices:
                        for call in choice.message.tool_calls or []:
                            call.id = str(uuid4())
                            translation = self.dialect.translate(
                                call, {tool.name for tool in tools}
                            )
                            if translation:
                                with Path(options["translation_log"]).open("a") as f:
                                    f.write(
                                        json.dumps(
                                            {"model": self.model_name, **translation}
                                        )
                                        + "\n"
                                    )
                    # Tool-less forced answers are returned to CT for its native retry/unscored ledger.
                    return output

            return LinuxArenaTinkerAPI

        registry.modelapi_register(make_api(monitoring), route)


def launch(options_path: Path, env_file: Path, arguments: list[str]) -> None:
    """Configure the bridge, bound the sandbox, then invoke CT's CLI."""
    import sys

    options = json.loads(options_path.read_text())
    dotenv = import_module("dotenv")
    dotenv.load_dotenv(env_file)
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not configured")
    os.environ.update(
        USE_OPENROUTER="0",
        SKIP_DOCKER_BUILD="1",
        CONTROL_TOWER_PRICING="off",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    register_provider(options)
    docker = import_module("control_tower.sandbox.docker")
    yaml = import_module("yaml")
    original = docker.get_compose_path_for_eval

    def bounded(compose_path: Path, **kwargs: Any) -> Path:
        # Validate before CT's original helper can even pull images.
        data = bounded_compose(
            yaml.safe_load(compose_path.read_text()), options["resources"]
        )
        safe_path = docker.write_temp_compose(
            compose_path, "linuxarena-resource-limits", data
        )
        return original(safe_path, **kwargs)

    setattr(docker, "get_compose_path_for_eval", bounded)
    sys.argv = ["ct", *arguments]
    next(iter(entry_points(group="console_scripts").select(name="ct"))).load()()
