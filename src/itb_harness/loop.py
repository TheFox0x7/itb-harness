"""Main observer/act loop.

One iteration is:

    screenshot -> model returns point actions -> execute click/hover -> screenshot

The model does not use tool calling in this mode.  It replies with a JSON list
of normalized points and a label, for example:

    [{"point_2d": [500, 250], "label": "click"}]

Points are processed in order, then a fresh screenshot is captured.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import ChatAgent, MistralAgent, ModelTurn
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
from .points import (
    PointAction,
    PointParseError,
    parse_point_actions,
    point_to_pixels,
)


SYSTEM_PROMPT = """\
You are controlling the game Into the Breach.

You receive screenshots of the game window. Use your vision to decide what to
do next. Each point you return will move the pointer to that point and perform
the action named by its label.

Return only a JSON array of point actions:

[
  {"point_2d": [500, 250], "label": "click"},
  {"point_2d": [750, 250], "label": "hover"}
]

Rules:
- point_2d is [x, y], normalized from 0 to 1000.
- x=0 is the left edge, x=1000 the right edge.
- y=0 is the top edge, y=1000 the bottom edge.
- label is either "click" or "hover".
- Actions are performed in the order you provide them.
- After the whole list is processed, a new screenshot will be provided.
- If unsure, return a single input to see the result.
- If an input fails, re-examine the screenshot before retrying. Do not repeat a
  failed action blindly.

Do not return prose, markdown, tool calls, or explanations. Return only the
JSON array.
"""


@dataclass(slots=True)
class LoopConfig:
    max_steps: int = 20
    max_actions_per_turn: int = 12
    max_images: int = 4
    action_delay: float = 0.15
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    model: str = "mistral-small-2603"


@dataclass(slots=True)
class LoopResult:
    steps: int
    actions: int
    frames: int
    run_dir: Path
    stopped: str
    final_text: str | None = None


@dataclass(frozen=True, slots=True)
class PointOutcome:
    index: int
    action: PointAction
    pixel: tuple[int, int]
    screen: tuple[int, int]
    ok: bool
    error: str | None = None


class Recorder:
    """Write frames, model turns, and point actions for one run."""

    def __init__(
        self,
        out_dir: Path,
        backend: CaptureBackend,
        input_backend: InputBackend,
        model: str,
        target: WindowInfo,
        config: LoopConfig,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.root = out_dir / f"{stamp}-agent"
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
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
                    "max_steps": config.max_steps,
                    "max_actions_per_turn": config.max_actions_per_turn,
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
        self._conversation.write(
            json.dumps({"step": step, "content": turn.content}) + "\n"
        )
        self._conversation.flush()

    def log_actions(
        self,
        step: int,
        outcomes: list[PointOutcome],
        frame_path: Path,
        next_frame_path: Path,
    ) -> None:
        for outcome in outcomes:
            record = {
                "step": step,
                "index": outcome.index,
                "label": outcome.action.label,
                "point_2d": outcome.action.point_2d,
                "pixel": list(outcome.pixel),
                "screen": list(outcome.screen),
                "ok": outcome.ok,
                "error": outcome.error,
                "frame": str(frame_path.relative_to(self.root)),
                "next_frame": str(next_frame_path.relative_to(self.root)),
            }
            self._actions.write(json.dumps(record) + "\n")
        self._actions.flush()

    def close(self) -> None:
        self._actions.close()
        self._conversation.close()


class HarnessLoop:
    """Screenshot -> point list -> click/hover -> screenshot."""

    def __init__(
        self,
        backend: CaptureBackend,
        target: WindowInfo,
        agent: ChatAgent,
        input_backend: InputBackend,
        config: LoopConfig | None = None,
    ) -> None:
        self.backend = backend
        self.target = target
        self.agent = agent
        self.input = input_backend
        self.config = config or LoopConfig()
        self.recorder = Recorder(
            self.config.out_dir,
            backend,
            input_backend,
            getattr(agent, "model", "unknown"),
            target,
            self.config,
        )

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.frame = self._capture()
        self.last_frame_path = self.recorder.frames_dir / "0001.png"
        self.messages.append(
            _user_image_message(
                self.frame,
                "Initial screenshot. It is your turn. Return a JSON list of "
                "point actions.",
            )
        )
        self.action_count = 0

    def _capture(self) -> Frame:
        frame = self.backend.capture_window(self.target.id)
        self.target = frame.window
        path = self.recorder.save_frame(frame)
        self.last_frame_path = path
        print(
            f"[frame {self.recorder.frame_index:04d}] "
            f"{frame.width}x{frame.height}  {frame.window.monitor or '?':6}  "
            f"{path}"
        )
        return frame

    def _prune_images(self) -> None:
        # Keep at least the newest frame, even if the config is set to 0.
        keep = max(1, self.config.max_images)
        image_messages = []
        for message in self.messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(chunk, dict) and chunk.get("type") == "image_url"
                for chunk in content
            ):
                image_messages.append(message)

        for message in image_messages[:-keep]:
            message["content"] = [
                {"type": "text", "text": "[older screenshot omitted]"}
            ]

    def _execute_actions(
        self, step: int, actions: list[PointAction]
    ) -> tuple[list[PointOutcome], Path]:
        frame = self.frame
        pre_action_frame = self.last_frame_path
        outcomes: list[PointOutcome] = []

        for index, action in enumerate(actions, start=1):
            px, py = point_to_pixels(action, frame.width, frame.height)
            sx = frame.window.x + px
            sy = frame.window.y + py
            try:
                if action.label == "click":
                    self.input.click(sx, sy)
                else:
                    self.input.hover(sx, sy)
                ok = True
                error = None
            except Exception as exc:
                ok = False
                error = str(exc)

            outcome = PointOutcome(
                index=index,
                action=action,
                pixel=(px, py),
                screen=(sx, sy),
                ok=ok,
                error=error,
            )
            outcomes.append(outcome)
            self.action_count += 1

            status = "ok" if ok else f"error={error}"
            print(
                f"[step {step:02d}.{index:02d}] {action.label} "
                f"{action.point_2d} -> pixel {px},{py} -> "
                f"screen {sx},{sy} ({status})"
            )

            if self.config.action_delay > 0:
                time.sleep(self.config.action_delay)

        # A list represents one action batch; capture once afterwards.
        self.frame = self._capture()
        self.recorder.log_actions(
            step, outcomes, pre_action_frame, self.last_frame_path
        )
        return outcomes, self.last_frame_path

    def run(self) -> LoopResult:
        stopped = "max_steps"
        final_text: str | None = None
        step = 0

        try:
            for step in range(1, self.config.max_steps + 1):
                turn = self.agent.complete(self.messages)
                self.recorder.log_turn(step, turn)
                if turn.content:
                    print(f"\n[model] {turn.content}")
                self.messages.append(
                    {"role": "assistant", "content": turn.content}
                )

                try:
                    actions = parse_point_actions(turn.content or "")
                except PointParseError as exc:
                    print(f"[parse error] {exc}")
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your response could not be parsed: {exc} "
                                "Return only a JSON array like "
                                '[{"point_2d": [500, 250], "label": "click"}].'
                            ),
                        }
                    )
                    continue

                if not actions:
                    stopped = "model_stopped"
                    final_text = turn.content
                    break

                if len(actions) > self.config.max_actions_per_turn:
                    print(
                        f"[parse error] {len(actions)} actions exceeds "
                        f"limit {self.config.max_actions_per_turn}"
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Return at most "
                                f"{self.config.max_actions_per_turn} actions "
                                "in one list."
                            ),
                        }
                    )
                    continue

                outcomes, next_frame_path = self._execute_actions(step, actions)
                self.messages.append(
                    _user_image_message(
                        self.frame,
                        f"Processed {len(outcomes)} input(s). "
                        "Here is the resulting screenshot. Return the next "
                        "JSON list of point actions.",
                    )
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


def _user_image_message(frame: Frame, text: str) -> dict[str, Any]:
    encoded = base64.b64encode(frame.png).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ],
    }


def _resolve_target(backend: CaptureBackend, spec: str) -> WindowInfo:
    return resolve_window(backend.list_windows(), spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itb-agent",
        description="Run the observer/act loop for Into the Breach.",
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
        help="Maximum model turns (default: 20).",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=12,
        help="Maximum point actions per model turn (default: 12).",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.15,
        help="Seconds to wait after each click/hover (default: 0.15).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Mistral model (default: MISTRAL_MODEL or mistral-small-2603).",
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
            agent = MistralAgent(model=args.model)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: could not initialize model client: {exc}", file=sys.stderr)
            return 2

        config = LoopConfig(
            max_steps=max(1, args.steps),
            max_actions_per_turn=max(1, args.max_actions),
            action_delay=max(0.0, args.action_delay),
            out_dir=args.out.expanduser(),
            model=agent.model,
        )
        loop = HarnessLoop(backend, target, agent, input_backend, config)
        print(f"Target: [{target.monitor or '?'}] {target.label}")
        print(f"Input:  {input_backend.name}")
        try:
            result = loop.run()
        except Exception as exc:
            print(f"error: loop failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"\nStopped: {result.stopped}  steps={result.steps}  "
            f"actions={result.actions}  frames={result.frames}"
        )
        print(f"Run output: {result.run_dir}")
        return 0
    finally:
        if input_backend is not None:
            input_backend.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
