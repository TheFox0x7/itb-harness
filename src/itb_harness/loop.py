"""Main observer/act loop.

The model is not given an automatic screenshot.  It has three tools:

    view_board()          capture and return the board
    click(point_2d)       click at screenshot pixel (x, y)
    inspect(point_2d)     hover, wait, capture and return a screenshot

The loop executes one tool call per model turn, feeds the tool result back, and
lets the model decide what to do next.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import ChatAgent, MistralAgent, ModelToolCall, ModelTurn
from .board_tools import (
    parse_inspect_args,
    parse_note_args,
    parse_point_args,
    parse_select_unit_args,
    parse_weapon_args,
    tool_schemas,
)
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


UNIT_KEYS = {1: "a", 2: "s", 3: "d", 4: "z", 5: "x"}

SYSTEM_PROMPT = """\
You are controlling the game Into the Breach.

You do not receive screenshots automatically. Use the tools supplied with the
request:

- view_board: capture and return the current game screenshot.
- click: move to a screenshot pixel coordinate and left-click.
- inspect: hover at a screenshot pixel coordinate, wait briefly, then capture
  and return a screenshot. Use with_details=true to hold Left Ctrl for
  detailed tooltips.
- attack_order: hold Left Alt and capture the attack-order preview.
- repair: press R to activate repair mode, then click the selected mech.
- weapon: press 1 or 2 to select a weapon, then click the target.
- select_unit: select unit 1-5 (A/S/D/Z/X).
- end_turn, undo_move, reset_turn, cancel: turn and cancellation controls.

Coordinates are [x, y] in screenshot pixels, with (0, 0) at the top-left and
the bottom-right at (width - 1, height - 1). Use the dimensions reported by
view_board. Call view_board before your first click, and call view_board again
after a click when you need to see the result. Inspect before committing an
attack when you are unsure of the result.

Use exactly one tool per turn. Do not describe an action in prose; call a tool.
If a tool returns an error, inspect the last screenshot and try a different
approach. Do not repeat a failed action blindly.

The game has 3 phases:
- Deployment where you need to place your units on tan spots. 
Pay attention to building placements and enemies to try and get early advantage.
After you deploy all your units click confirm to deploy them. Then the enemy takes it's initial turn.
You can ONLY place the units on the tan spots. To deploy a unit you must click on the tile.
Clicking the tile again will swap units with the next one that is to be deployed or will allow you to change placement.
Yellow arrow under a unit means the unit will be deployed to that spot. A refresh icon indicates that mech will be swapped with one that is about to be deployed.

- Enemy turn. If you see that information in the screenshot you should wait and take another. You can't do anything while enemy takes it's turn.
- Your turn. Now you get to move your units. Each unit can move and attack in the same turn but an attack ends the turn.
Units can have multiple abilities to use on the lower bar of the screen. Inspect them to learn more and plan to use the attacks optimally.

Use notes tool to note down things you learned or misunderstood about instructions so future iterations can be adjusted.

Your primary goal is to protect buildings and accomplish side objectives. Sometimes sacrifice of unit health or entire unit can be optimal,
but it will mean you lose the pilot and all the perks they had if mech is destroyed. Your power grid shows how many times you can lose a building in the run.
You lose the game it it reaches 0 bars. Grid defense is a chance to mitigate the damage to the grid completely, however it works per attack.
"""


@dataclass(slots=True)
class LoopConfig:
    max_steps: int = 20
    max_images: int = 4
    action_delay: float = 0.15
    show_thinking: bool = False
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

    def __init__(self) -> None:
        self.started = False

    def write(self, text: str) -> None:
        if not text:
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

        print(f"Target  [{target.monitor or '?'}] {target.label}")
        print(f"Capture {backend.name}")
        print(f"Input   {input_backend.name}")
        if notes is not None:
            print(f"Notes   {notes.path} ({notes.char_count} chars loaded)")
        print(f"Output  {self.recorder.root}")

        system_prompt = SYSTEM_PROMPT
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

    def _capture(self) -> Frame:
        frame = self.backend.capture_window(self.target.id)
        self.target = frame.window
        self.frame = frame
        path = self.recorder.save_frame(frame)
        self.last_frame_path = path
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

    def _require_frame(self) -> Frame:
        if self.frame is None or self.last_frame_path is None:
            raise ValueError("Call view_board before click or inspect.")
        return self.frame

    def _pause(self) -> None:
        if self.config.action_delay > 0:
            time.sleep(self.config.action_delay)

    def _validate_point(self, frame: Frame, x: int, y: int) -> None:
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            raise ValueError(
                f"Point ({x}, {y}) is outside the screenshot "
                f"{frame.width}x{frame.height}."
            )

    def _execute_tool(
        self, call: ModelToolCall
    ) -> tuple[ToolResult, Path | None]:
        try:
            if call.name == "note":
                if self.notes is None:
                    return (
                        ToolResult(
                            ok=False,
                            error="notes_disabled",
                            message="Notes are not enabled for this run.",
                        ),
                        None,
                    )
                text = parse_note_args(call.arguments)
                self.notes.append(text, run_id=self.recorder.root.name)
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Saved note for future runs "
                            f"({len(text)} chars)."
                        ),
                    ),
                    None,
                )

            if call.name == "view_board":
                frame = self._capture()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Current board screenshot attached "
                            f"({frame.width}x{frame.height})."
                        ),
                        image_png=frame.png,
                    ),
                    self.last_frame_path,
                )

            if call.name == "click":
                frame = self._require_frame()
                x, y = parse_point_args(call.arguments)
                self._validate_point(frame, x, y)
                sx = frame.window.x + x
                sy = frame.window.y + y
                self.input.click(sx, sy)
                self.action_count += 1
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Clicked screenshot pixel ({x}, {y}) "
                            f"at screen ({sx}, {sy})."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "inspect":
                frame = self._require_frame()
                x, y, with_details = parse_inspect_args(call.arguments)
                self._validate_point(frame, x, y)
                sx = frame.window.x + x
                sy = frame.window.y + y
                self.input.hover(sx, sy)
                if with_details:
                    self.input.key_down("ctrl")
                try:
                    self._pause()
                    frame = self._capture()
                finally:
                    if with_details:
                        self.input.key_up("ctrl")
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Hovered screenshot pixel ({x}, {y}) and "
                            f"captured a {frame.width}x{frame.height} "
                            "screenshot"
                            + (" with details." if with_details else ".")
                        ),
                        image_png=frame.png,
                    ),
                    self.last_frame_path,
                )

            if call.name == "attack_order":
                self.input.key_down("alt")
                try:
                    self._pause()
                    frame = self._capture()
                finally:
                    self.input.key_up("alt")
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Attack-order preview attached "
                            f"({frame.width}x{frame.height})."
                        ),
                        image_png=frame.png,
                    ),
                    self.last_frame_path,
                )

            if call.name == "repair":
                self.input.key("r")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Repair mode activated. Click the selected mech "
                            "as the repair target."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "weapon":
                number = parse_weapon_args(call.arguments)
                self.input.key(str(number))
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Weapon {number} selected. Inspect first to "
                            "preview possible results, then click the target."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "select_unit":
                number = parse_select_unit_args(call.arguments)
                self.input.key(UNIT_KEYS[number])
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Selected unit {number} with key "
                            f"{UNIT_KEYS[number]!r}."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "end_turn":
                self.input.key("space")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message="End-turn key pressed.",
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "undo_move":
                self.input.key("shift")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Undo-move key pressed. Only moves can be undone; "
                            "attacks are permanent."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "reset_turn":
                self.input.key("backspace")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Reset-turn key pressed. This is available at "
                            "most once per mission and may be unavailable."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "cancel":
                self.input.right_click()
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message="Right-clicked to cancel the current selection.",
                        state_changed=True,
                    ),
                    None,
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
            message = str(exc)
            error = (
                "no_frame"
                if message.startswith("Call view_board")
                else "invalid_arguments"
            )
            return (
                ToolResult(ok=False, error=error, message=message),
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

    def run(self) -> LoopResult:
        stopped = "max_steps"
        final_text: str | None = None
        step = 0

        try:
            for step in range(1, self.config.max_steps + 1):
                print(f"\n[turn {step}/{self.config.max_steps}]")
                printer = ThinkingPrinter()
                try:
                    turn = self.agent.complete(
                        self.messages,
                        tools=tool_schemas(include_note=self.notes is not None),
                        on_thinking=printer.write,
                    )
                finally:
                    printer.close()

                self.recorder.log_turn(step, turn)
                if turn.thinking and not printer.started:
                    if self.config.show_thinking:
                        print(textwrap.indent(turn.thinking.strip(), "  "))
                    else:
                        print(f"  thinking saved -> thinking/{step:04d}.txt")
                if turn.usage:
                    cached = turn.usage.get("cached_tokens")
                    prompt_tokens = turn.usage.get("prompt_tokens")
                    if cached is not None:
                        print(
                            f"  cache: {cached}/"
                            f"{prompt_tokens if prompt_tokens is not None else '?'} "
                            "prompt tokens"
                        )

                self.messages.append(_assistant_message(turn))

                if not turn.tool_calls:
                    print("  model returned no tool call; stopping")
                    stopped = "model_stopped"
                    final_text = turn.content
                    break

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

                source_frame = self.last_frame_path
                result, next_frame = self._execute_tool(primary)
                self.messages.append(
                    result.as_tool_message(primary.id, primary.name)
                )
                self.recorder.log_tool_call(
                    step, primary, result, next_frame, source_frame
                )

                if result.ok:
                    print(f"  tool {primary.name}: {result.message}")
                else:
                    print(
                        f"  tool {primary.name}: error={result.error} "
                        f"{result.message}"
                    )
                self._prune_images()
        except KeyboardInterrupt:
            stopped = "interrupted"
        finally:
            self.recorder.close()

        return LoopResult(
            steps=step,
            actions=self.action_count,
            frames=self.recorder.frame_index,
            run_dir=self.recorder.root,
            stopped=stopped,
            final_text=final_text,
        )


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
        help="Pointer input backend: stub (default) or x11.",
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
        help="Maximum model turns/tool calls (default: 20).",
    )
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
        default=0.15,
        help="Seconds to wait after clicks, keys, and hovers (default: 0.15).",
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
        help="Enable the optional note tool and load persistent notes.",
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
            show_thinking=args.show_thinking,
            out_dir=args.out.expanduser(),
            model=agent.model,
        )
        loop = HarnessLoop(
            backend, target, agent, input_backend, config, notes=notes
        )
        try:
            result = loop.run()
        except Exception as exc:
            print(f"error: loop failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"\nStopped: {result.stopped}  steps={result.steps}  "
            f"clicks={result.actions}  frames={result.frames}"
        )
        print(f"Run output: {result.run_dir}")
        return 0
    finally:
        if input_backend is not None:
            input_backend.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
