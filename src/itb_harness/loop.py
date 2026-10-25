"""Main observation and movement loop for Helltaker.

The model sees screenshots and can move one or more tiles at a time with the
four arrow-key movement tools.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import ChatAgent, MistralAgent, ModelToolCall, ModelTurn
from .board_tools import parse_move_args, tool_schemas
from .capture import (
    BackendUnavailable,
    CaptureBackend,
    CaptureError,
    Frame,
    WindowInfo,
    get_backend,
    resolve_window,
)
from .cli import format_windows
from .input import InputBackend, get_input_backend
from .notes import NotesStore
from .tools import ToolResult


INPUT_TOOLS = {"move_up", "move_down", "move_left", "move_right"}
MOVE_KEYS = {
    "move_up": "up",
    "move_down": "down",
    "move_left": "left",
    "move_right": "right",
}

SYSTEM_PROMPT = """\
You are controlling Helltaker, a grid-based puzzle game.

You do not receive screenshots automatically. Use view_board to capture the
current game screen.

Movement tools:
- move_up: press the Up arrow.
- move_down: press the Down arrow.
- move_left: press the Left arrow.
- move_right: press the Right arrow.

Each movement tool accepts a `steps` argument. Use steps > 1 to move several
tiles in one batch instead of making several tool calls. Movement is
tile-based: one step should move exactly one tile unless blocked by a wall,
object, or the edge of the room.

Call view_board first. After moving, call view_board again when you need to
see the result or re-orient yourself. Use exactly one tool per turn. Do not
describe an action in prose; call a tool.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


@dataclass(slots=True)
class LoopConfig:
    max_steps: int = 20
    max_images: int = 4
    action_delay: float = 0.5
    move_delay: float = 0.1
    show_thinking: bool = False
    console: bool = True
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    model: str = "mistral-large-4"


@dataclass(slots=True)
class LoopResult:
    steps: int
    actions: int
    frames: int
    run_dir: Path
    stopped: str
    final_text: str | None = None


class Recorder:
    """Write frames, model turns, and tool results for one run."""

    def __init__(
        self,
        out_dir: Path,
        backend: CaptureBackend,
        input_backend: InputBackend,
        model: str,
        cache_key: str | None,
        notes: NotesStore | None,
        target: WindowInfo,
        config: LoopConfig,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.root = out_dir / f"{stamp}-agent"
        self.frames_dir = self.root / "frames"
        self.thinking_dir = self.root / "thinking"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.thinking_dir.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0

        self.actions_path = self.root / "actions.jsonl"
        self.conversation_path = self.root / "conversation.jsonl"
        self._actions = self.actions_path.open("a", encoding="utf-8")
        self._conversation = self.conversation_path.open("a", encoding="utf-8")

        (self.root / "run.json").write_text(
            json.dumps(
                {
                    "backend": backend.name,
                    "input": input_backend.name,
                    "model": model,
                    "prompt_cache_key": cache_key,
                    "notes_file": str(notes.path) if notes is not None else None,
                    "max_steps": config.max_steps,
                    "max_images": config.max_images,
                    "action_delay": config.action_delay,
                    "move_delay": config.move_delay,
                    "started_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "target": asdict(target),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def save_frame(self, frame: Frame) -> Path:
        self.frame_index += 1
        path = self.frames_dir / f"{self.frame_index:04d}.png"
        path.write_bytes(frame.png)
        return path

    def log_turn(self, step: int, turn: ModelTurn) -> None:
        record: dict[str, Any] = {
            "step": step,
            "content": turn.content,
            "thinking": turn.thinking,
            "usage": turn.usage,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": _jsonable(call.arguments),
                }
                for call in turn.tool_calls
            ],
        }
        if turn.thinking:
            path = self.thinking_dir / f"{step:04d}.txt"
            path.write_text(turn.thinking, encoding="utf-8")
            record["thinking_file"] = str(path.relative_to(self.root))
        self._conversation.write(json.dumps(record) + "\n")
        self._conversation.flush()

    def log_tool_call(
        self,
        step: int,
        call: ModelToolCall,
        result: ToolResult,
        frame_path: Path | None,
        source_frame_path: Path | None,
    ) -> None:
        record = {
            "step": step,
            "tool": call.name,
            "arguments": _jsonable(call.arguments),
            "ok": result.ok,
            "error": result.error,
            "message": result.message,
            "state_changed": result.state_changed,
            "pixel": list(result.pixel) if result.pixel is not None else None,
            "frame": (
                str(frame_path.relative_to(self.root))
                if frame_path is not None
                else None
            ),
            "source_frame": (
                str(source_frame_path.relative_to(self.root))
                if source_frame_path is not None
                else None
            ),
        }
        self._actions.write(json.dumps(record) + "\n")
        self._actions.flush()

    def close(self) -> None:
        self._actions.close()
        self._conversation.close()


class ThinkingPrinter:
    """Stream reasoning chunks to the terminal as they arrive."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started = False

    def write(self, text: str) -> None:
        if not self.enabled or not text:
            return
        if not self.started:
            print("  thinking:", flush=True)
            self.started = True
        sys.stdout.write(text.replace("\n", "\n    "))
        sys.stdout.flush()

    def close(self) -> None:
        if self.started:
            print()


class HarnessLoop:
    """Tool-driven observe/act loop."""

    def __init__(
        self,
        backend: CaptureBackend,
        target: WindowInfo,
        agent: ChatAgent,
        input_backend: InputBackend,
        config: LoopConfig | None = None,
        notes: NotesStore | None = None,
    ) -> None:
        self.backend = backend
        self.target = target
        self.agent = agent
        self.input = input_backend
        self.notes = notes
        self.config = config or LoopConfig()
        self.recorder = Recorder(
            self.config.out_dir,
            backend,
            input_backend,
            getattr(agent, "model", "unknown"),
            getattr(agent, "prompt_cache_key", None),
            notes,
            target,
            self.config,
        )

        self.frame: Frame | None = None
        self.last_frame_path: Path | None = None
        self.action_count = 0
        self.step_index = 0
        self.final_text: str | None = None
        self._closed = False
        self.event_callback: Callable[[dict[str, Any]], None] | None = None

        if self.config.console:
            print(f"Target  [{target.monitor or '?'}] {target.label}")
            print(f"Capture {backend.name}")
            print(f"Input   {input_backend.name}")
            if notes is not None:
                print(f"Notes   {notes.path} ({notes.char_count} chars loaded)")
            print(f"Output  {self.recorder.root}")

        system_prompt = build_system_prompt()
        if input_backend.name == "stub":
            system_prompt += (
                "\n\nInput is in stub/dry-run mode: actions are recorded but "
                "NOT sent to the game, so the board will not change."
            )
        if notes is not None:
            rendered_notes = notes.render_for_prompt()
            if rendered_notes:
                system_prompt += "\n\n" + rendered_notes

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Start by calling view_board. Use one tool per turn."
                ),
            },
        ]

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.event_callback is not None:
            self.event_callback({"type": event_type, **data})

    def _capture(self) -> Frame:
        frame = self.backend.capture_window(self.target.id)
        self.target = frame.window
        self.frame = frame
        path = self.recorder.save_frame(frame)
        self.last_frame_path = path
        self._emit(
            "frame",
            path=str(path.relative_to(self.recorder.root)),
            width=frame.width,
            height=frame.height,
        )
        if self.config.console:
            print(
                f"  frame {self.recorder.frame_index:04d}  "
                f"{frame.width}x{frame.height}  "
                f"[{frame.window.monitor or '?'}]  "
                f"{path.relative_to(self.recorder.root)}"
            )
        return frame

    def _prune_images(self) -> None:
        # Keep the latest screenshots and replace older tool-message images
        # with text.  This changes the prompt prefix and therefore invalidates
        # prompt cache entries from that point onward; --max-images 0 keeps the
        # full append-only history for better caching.
        if self.config.max_images <= 0:
            return

        image_messages = []
        for message in self.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(chunk, dict) and chunk.get("type") == "image_url"
                for chunk in content
            ):
                image_messages.append(message)

        for message in image_messages[: -self.config.max_images]:
            message["content"] = json.dumps(
                {"ok": True, "message": "[older screenshot omitted]"}
            )

    def _pause(self) -> None:
        if self.config.action_delay > 0:
            time.sleep(self.config.action_delay)

    def _execute_tool(
        self, call: ModelToolCall
    ) -> tuple[ToolResult, Path | None]:
        try:
            if call.name == "view_board":
                frame = self._capture()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Current screenshot attached "
                            f"({frame.width}x{frame.height})."
                        ),
                        image_png=frame.png,
                    ),
                    self.last_frame_path,
                )

            key = MOVE_KEYS.get(call.name)
            if key is not None:
                steps = parse_move_args(call.arguments)
                for index in range(steps):
                    self.input.key(key)
                    if (
                        index + 1 < steps
                        and self.config.move_delay > 0
                    ):
                        time.sleep(self.config.move_delay)
                self.action_count += steps
                self._pause()
                frame = self._capture()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Moved {key} {steps} "
                            f"{'step' if steps == 1 else 'steps'}. "
                            f"Screenshot attached "
                            f"({frame.width}x{frame.height})."
                        ),
                        image_png=frame.png,
                        state_changed=True,
                    ),
                    self.last_frame_path,
                )

            return (
                ToolResult(
                    ok=False,
                    error="unknown_tool",
                    message=f"Unknown tool {call.name!r}.",
                ),
                None,
            )
        except ValueError as exc:
            return (
                ToolResult(
                    ok=False,
                    error="invalid_arguments",
                    message=str(exc),
                ),
                None,
            )
        except Exception as exc:
            return (
                ToolResult(
                    ok=False,
                    error="tool_failed",
                    message=str(exc),
                ),
                None,
            )

    def step_once(self) -> bool:
        """Run one model turn and at most one tool call.

        Returns False when the model stops calling tools.
        """

        self.step_index += 1
        step = self.step_index
        if self.config.console:
            print(f"\n[turn {step}]")
        self._emit("status", phase="model_waiting", step=step)
        printer = ThinkingPrinter(enabled=self.config.console)

        def on_thinking(text: str) -> None:
            printer.write(text)
            self._emit("thinking", text=text, step=step)

        def on_content(text: str) -> None:
            self._emit("content", text=text, step=step)

        try:
            turn = self.agent.complete(
                self.messages,
                tools=tool_schemas(),
                on_thinking=on_thinking,
                on_content=on_content,
            )
        finally:
            printer.close()

        self.recorder.log_turn(step, turn)
        if self.config.console and turn.thinking and not printer.started:
            if self.config.show_thinking:
                print(textwrap.indent(turn.thinking.strip(), "  "))
            else:
                print(f"  thinking saved -> thinking/{step:04d}.txt")
        if self.config.console and turn.usage:
            cached = turn.usage.get("cached_tokens")
            prompt_tokens = turn.usage.get("prompt_tokens")
            if cached is not None:
                print(
                    f"  cache: {cached}/"
                    f"{prompt_tokens if prompt_tokens is not None else '?'} "
                    "prompt tokens"
                )

        self.messages.append(_assistant_message(turn))

        self._emit(
            "turn",
            step=step,
            usage=turn.usage,
            content=turn.content,
            tool_call_count=len(turn.tool_calls),
        )

        if not turn.tool_calls:
            if self.config.console:
                print("  model returned no tool call; stopping")
            self._emit("status", phase="model_stopped", step=step)
            self.final_text = turn.content
            return False

        primary = turn.tool_calls[0]
        for extra in turn.tool_calls[1:]:
            extra_result = ToolResult(
                ok=False,
                error="one_tool_per_turn",
                message="Call exactly one tool per turn.",
            )
            self.messages.append(
                extra_result.as_tool_message(extra.id, extra.name)
            )
            self.recorder.log_tool_call(
                step,
                extra,
                extra_result,
                None,
                self.last_frame_path,
            )

        self._emit(
            "status", phase="tool", step=step, tool=primary.name
        )
        source_frame = self.last_frame_path
        result, next_frame = self._execute_tool(primary)
        if (
            self.input.name == "stub"
            and primary.name in INPUT_TOOLS
            and result.ok
        ):
            result = replace(
                result,
                state_changed=False,
                message=(
                    f"{result.message} "
                    "(stub input: action was recorded but not sent)"
                ),
            )
        self.messages.append(
            result.as_tool_message(primary.id, primary.name)
        )
        self.recorder.log_tool_call(
            step, primary, result, next_frame, source_frame
        )
        self._emit(
            "tool_result",
            step=step,
            tool=primary.name,
            ok=result.ok,
            error=result.error,
            message=result.message,
            pixel=list(result.pixel) if result.pixel is not None else None,
            frame=(
                str(next_frame.relative_to(self.recorder.root))
                if next_frame is not None
                else None
            ),
        )
        self._emit("status", phase="turn_done", step=step)

        if self.config.console:
            if result.ok:
                print(f"  tool {primary.name}: {result.message}")
            else:
                print(
                    f"  tool {primary.name}: error={result.error} "
                    f"{result.message}"
                )
        self._prune_images()
        return True

    def run(self, max_steps: int | None = None) -> LoopResult:
        limit = self.config.max_steps if max_steps is None else max_steps
        stopped = "max_steps"

        try:
            while limit is None or limit <= 0 or self.step_index < limit:
                if not self.step_once():
                    stopped = "model_stopped"
                    break
        except KeyboardInterrupt:
            stopped = "interrupted"
        finally:
            self.close()

        return LoopResult(
            steps=self.step_index,
            actions=self.action_count,
            frames=self.recorder.frame_index,
            run_dir=self.recorder.root,
            stopped=stopped,
            final_text=self.final_text,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.recorder.close()


def _assistant_message(turn: ModelTurn) -> dict[str, Any]:
    tool_calls = []
    for call in turn.tool_calls:
        arguments = call.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(_jsonable(arguments))
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }
        )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _resolve_target(backend: CaptureBackend, spec: str) -> WindowInfo:
    return resolve_window(backend.list_windows(), spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itb-agent",
        description="Run the tool-driven observer/act loop.",
    )
    parser.add_argument(
        "--backend", default="auto", help="Capture backend: auto, kwin, x11."
    )
    parser.add_argument(
        "-w",
        "--window",
        help="Target window: index, id, app id, or title substring.",
    )
    parser.add_argument(
        "--input",
        default="stub",
        help="Input backend: stub (default) or x11.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="Directory for run output (default: runs).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help=(
            "Headless maximum model turns/tool calls (default: 20). "
            "Ignored with --web."
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the interactive browser UI instead of running headless.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-images",
        type=int,
        default=4,
        help=(
            "Screenshots kept in conversation history; 0 keeps everything "
            "(better prompt-cache reuse, larger context)."
        ),
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.5,
        help=(
            "Seconds to wait after a movement batch before capturing "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--move-delay",
        type=float,
        default=0.1,
        help=(
            "Seconds between individual key presses in a movement batch "
            "(default: 0.1)."
        ),
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream model thinking to the terminal (default: on).",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help=(
            "With --no-stream, print the saved thinking after the response. "
            "Streaming mode already prints it live."
        ),
    )
    parser.add_argument(
        "--notes",
        action="store_true",
        help="Load persistent notes into the prompt.",
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        default=Path("notes.md"),
        help="Persistent notes file (default: notes.md).",
    )
    parser.add_argument(
        "--notes-max-chars",
        type=int,
        default=6000,
        help="Maximum note characters loaded into the prompt (default: 6000).",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=180.0,
        help="Mistral request timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--cache-key",
        default=None,
        help=(
            "Stable Mistral prompt_cache_key; defaults to "
            "MISTRAL_PROMPT_CACHE_KEY or a per-run UUID."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Mistral model (default: MISTRAL_MODEL or mistral-large-4).",
    )
    parser.add_argument(
        "--list",
        dest="list_windows",
        action="store_true",
        help="List windows and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        backend = get_backend(args.backend)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    input_backend: InputBackend | None = None
    try:
        if args.list_windows:
            print(format_windows(backend.list_windows()))
            return 0

        if not args.window:
            print("error: --window is required", file=sys.stderr)
            print(format_windows(backend.list_windows()), file=sys.stderr)
            return 2

        try:
            target = _resolve_target(backend, args.window)
        except (ValueError, CaptureError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        try:
            input_backend = get_input_backend(args.input)
        except BackendUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        try:
            agent = MistralAgent(
                model=args.model,
                prompt_cache_key=args.cache_key,
                stream=args.stream,
                timeout_ms=max(1, int(args.model_timeout * 1000)),
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: could not initialize model client: {exc}", file=sys.stderr)
            return 2

        notes = (
            NotesStore(
                args.notes_file.expanduser(),
                max_chars=args.notes_max_chars,
            )
            if args.notes
            else None
        )

        config = LoopConfig(
            max_steps=max(1, args.steps),
            max_images=args.max_images,
            action_delay=max(0.0, args.action_delay),
            move_delay=max(0.0, args.move_delay),
            show_thinking=args.show_thinking,
            console=not args.web,
            out_dir=args.out.expanduser(),
            model=agent.model,
        )
        if input_backend.name == "stub":
            print(
                "warning: input backend is 'stub'; keys are recorded but "
                "NOT sent to the game. Use --input x11 for real input.",
                file=sys.stderr,
            )

        loop = HarnessLoop(
            backend, target, agent, input_backend, config, notes=notes
        )

        if args.web:
            import uvicorn

            from .session import SessionRunner
            from .web.server import create_app

            runner = SessionRunner(loop)
            app = create_app(config.out_dir, live=runner)
            runner.start()
            print(f"Live UI: http://{args.host}:{args.port}")
            print("Manual step mode; press Step in the browser to begin.")
            try:
                uvicorn.run(
                    app,
                    host=args.host,
                    port=args.port,
                    log_level="info",
                )
            finally:
                runner.stop()
                runner.join(timeout=5.0)
            return 0

        try:
            result = loop.run()
        except Exception as exc:
            print(f"error: loop failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"\nStopped: {result.stopped}  steps={result.steps}  "
            f"moves={result.actions}  frames={result.frames}"
        )
        print(f"Run output: {result.run_dir}")
        return 0
    finally:
        if input_backend is not None:
            input_backend.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
