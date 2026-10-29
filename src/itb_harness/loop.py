"""Main observation and movement loop for Helltaker.

The model sees screenshots and can move one tile at a time with the four
arrow-key movement tools.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import textwrap
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import ChatAgent, MistralAgent, ModelToolCall, ModelTurn
from .board_tools import (
    apply_move_to_level_map,
    format_level_map,
    parse_level_map_args,
    tool_schemas,
)
from .grid_overlay import annotate_helltaker_grid
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


INPUT_TOOLS = {
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "reset_level",
}
MOVE_KEYS = {
    "move_up": "up",
    "move_down": "down",
    "move_left": "left",
    "move_right": "right",
}


def _frames_differ(png_a: bytes, png_b: bytes, threshold: float = 0.005) -> bool:
    """Return True when two PNG screenshots differ enough to indicate a state change.

    Compares mean absolute pixel difference.  A blocked move (character did
    not move, no animation) produces nearly identical frames; a successful
    move changes the character position and possibly the step counter.
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image

    try:
        img_a = Image.open(BytesIO(png_a)).convert("RGB")
        img_b = Image.open(BytesIO(png_b)).convert("RGB")
    except Exception:
        return True  # can't compare → assume changed

    if img_a.size != img_b.size:
        return True

    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).mean()
    # Normalise to 0..1 range (max pixel diff is 255).
    return bool(diff / 255.0 > threshold)

SYSTEM_PROMPT = """\
You are controlling Helltaker, a grid-based puzzle game.

You do not receive screenshots automatically. Use view_board to capture the
current game screen.

Screenshots include a grid overlay: 68x72 pixel cells, up to 9 columns centered
on the screen, anchored to the red floor.

Tools:
- move_up: press the Up arrow.
- move_down: press the Down arrow.
- move_left: press the Left arrow.
- move_right: press the Right arrow.
- reset_level: press R to restart the current level and clear the level map.
- level_map: create or update your own text map of the level. Pass `rows`
  as a list of equal-width rows, for example ["[X] [X] [H]", "[X] [.] [D]"].
  Suggested symbols: X=wall/blocked, .=open floor, H=Helltaker, S=skeleton,
  T=spike trap, D=demon, R=rock, ?=unknown. Use `updates` to change individual cells
  or `clear` to remove the map. It does not change the game.

Each movement tool moves exactly one tile. Movement is tile-based: a move
should advance one tile unless blocked by a wall, object, or the edge of the
room.

Game rules:
- Skeleton knights can be kicked by moving into them. If kicked into the level
  edge, they are destroyed. You may see bones on the screenshot or it might be missing from the next image.
  They are not a threat to you, but they will delay your movement and make you waste steps.
- Kicking skeletons does not cascade and there usually is a better way than kicking every single one.
- Spikes are marked by white or black spots on the floor which alternate between hidden and showing as you move.
  They aren't in the early levels so you don't need to be cautious of them.
- Black tiles are edges and cannot be moved into. The map is grid based but it is not regular.
- Many obstacles are pushable by moving into them.
- Assume you can only move on the red space. Everything else is blocked. Rocks on the playing field might also be red and impassible
  but they are distinct from flat ground you can navigate.
- Moving into the wall or edges is not a valid move and will not consume steps. 
  However it is wasting your time so it is recommended for you to reexamine if your move consumed steps and if not, change strategy.
- Pillars are usually either obstacles or decoration. They aren't interactive and you can ignore them.
- White flashes on black screen means you ran out of steps and lost. Next call to view screen will return level as it was after reset.
- X on the left counter means you are on your last move. This usually means you're about to lose the game unless your next move brings you in front of the demon.

Do not be hesitant to test things out. You're not timed or graded on the task.
Your goal is to beat the level given and have fun doing it.
Trial and error are part of the learning experience and no one expects you to beat the level on your first try,
however you are encouraged to learn from your attempts on what worked well what did not.
Pay attention to your previous steps. User can interject and help you on your way.

Call view_board first. After moving, call view_board again when you need to
see the result or re-orient yourself. Do not describe an action in prose; call a tool.

Use the map to navigate and plan your moves. The map autoupdates on successful moves but does not take everything into account.
If your move is blocked but map has no obstacle that would suggest it, it means you mapped it out incorrectly. Look at it again instead of repeating.

=== MANDATORY WORKFLOW ===

1. PLAN BEFORE MOVING. After your first view_board, do NOT move yet.
   Instead, analyze the board and answer these questions in your thinking:
   - Where is the demon (the goal)?
   - Where are you (Helltaker, the character in the white suit)?
   - Where are the skeletons, spike traps, and other obstacles?
   - What is the shortest path from you to the demon?
   - How many moves does that path need? Do you have enough steps?
   Only start moving after you have a complete plan.

2. MARK THE BOARD. After analyzing use level_map once you understand the layout, then update it
   after moves or new observations so it stays accurate. The current map is
   included with subsequent tool results.

3. CHECK MOVE RESULTS. After every movement tool, the result tells you
   whether the move was BLOCKED (no step consumed) or consumed a step.
   - If BLOCKED: do NOT repeat the same move. Immediately try a different
     direction or call view_board to re-assess. Repeating a blocked move
     wastes a turn.
   - If the step counter decreased: the move worked. Continue your plan.

4. AFTER RESET, CHANGE STRATEGY. If you call reset_level, you MUST try a
   different approach than your previous attempt. If your last attempt went
   left-then-up, this attempt should go right-then-up, or push skeletons
   differently. Do not repeat the same moves after a reset.
   Note: after reset_level, the game may briefly show a title/loading
   screen. Call view_board once more — the level will appear.

5. OUTPUT ONLY TOOL CALLS. Your entire response should be a single tool
   call. No prose, no narration, no fake browser/automation output, no
   descriptions of what you are doing. Just the tool call.

6. KEEP THINKING CONCISE. State your plan in 2-3 sentences, then act.
   Do not re-analyze the same board state more than once. If you already
   know the layout, use that knowledge instead of re-examining it.

=== STRATEGIC GUIDANCE ===

These are general puzzle-solving principles. Apply them to any level:

- STUDY THE BOARD GEOMETRY. The map is not always a straight line from
  start to goal. Look for corridors, gaps, and alternative routes around
  obstacles. A wall of enemies does not always mean you must fight through
  it — sometimes the path goes around.

- DISTINGUISH OBSTACLE TYPES. Some things block movement (walls, edges,
  solid objects). Some things can be pushed or kicked out of the way
  (skeletons, pushable blocks). Some things are decoration and can be
  ignored. Observe what happens when you interact with each type.

- KICKING IS A TOOL, NOT A WEAPON. Kicking a skeleton moves it one tile
  and does not consume extra steps beyond the kick itself. Use kicks
  strategically: to clear a path, to push an enemy off an edge, or to
  make space for yourself. Kicking every enemy is usually wasteful.

- THE DEMON IS THE GOAL. She is usually at a fixed position on the map.
  Plan your route to reach the tile adjacent to her. Sometimes the direct
  path is blocked and you need to approach from a different direction.

- COUNT YOUR STEPS. Before committing to a path, estimate how many moves
  it needs. If it exceeds your remaining steps, find a shorter route.

- LEARN FROM BLOCKED MOVES. A blocked move tells you something about the
  board: there is a wall, edge, or solid object in that direction. Use
  this information to update your mental map. Do not waste turns probing
  the same blocked direction repeatedly.

- EXPLORE BEFORE COMMITTING. If you are unsure about a path, take one
  step to test it, then view_board to see the result. Do not blindly
  commit to a multi-move plan without checking intermediate states.

- RESET IS A LAST RESORT. Only reset when you have no better option.
  Before resetting, ask yourself: what did I learn? Can I use that
  knowledge to try something different without resetting?
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def _reference_message() -> dict[str, Any] | None:
    """Build the user message containing visual hazard references."""

    references = (
        (
            "Spikes - extended",
            "bright white/bone spikes protruding from the floor.",
            "spikes_extended.png",
        ),
        (
            "Spikes - hidden",
            "dark slots/holes in the red floor; the spikes are retracted "
            "but still dangerous.",
            "spikes_hidden.png",
        ),
        (
            "Rock",
            "large pink cracked boulder occupying a tile; it blocks movement.",
            "rock.png",
        ),
    )
    directory = Path(__file__).parent / "references"
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Reference images for Helltaker hazards. Use these to "
                "recognize them in screenshots:"
            ),
        }
    ]
    for label, description, filename in references:
        path = directory / filename
        if not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "text", "text": f"{label}: {description}"}
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded}",
                },
            }
        )

    if len(content) == 1:
        return None
    return {"role": "user", "content": content}


@dataclass(slots=True)
class LoopConfig:
    max_steps: int = 20
    max_images: int = 4
    action_delay: float = 0.5
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
        self.guidance_path = self.root / "guidance.jsonl"
        self.level_map_path = self.root / "level_map.jsonl"
        self._actions = self.actions_path.open("a", encoding="utf-8")
        self._conversation = self.conversation_path.open("a", encoding="utf-8")
        self._guidance_log = self.guidance_path.open("a", encoding="utf-8")
        self._level_map_log = self.level_map_path.open("a", encoding="utf-8")

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

    def log_guidance(self, step: int, text: str) -> None:
        record = {"step": step, "text": text}
        self._guidance_log.write(json.dumps(record) + "\n")
        self._guidance_log.flush()

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
            "blocked": result.blocked,
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

    def log_level_map(
        self, step: int, rows: list[list[str]] | None
    ) -> None:
        record = {"step": step, "rows": rows}
        self._level_map_log.write(json.dumps(record) + "\n")
        self._level_map_log.flush()

    def close(self) -> None:
        self._actions.close()
        self._conversation.close()
        self._guidance_log.close()
        self._level_map_log.close()


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
        self._prev_frame_png: bytes | None = None
        self.action_count = 0
        self.step_index = 0
        self.final_text: str | None = None
        self._closed = False
        self.event_callback: Callable[[dict[str, Any]], None] | None = None
        self.level_map: list[list[str]] | None = None
        self._move_history: list[dict[str, object]] = []
        self._guidance: deque[str] = deque()
        self._guidance_lock = threading.Lock()

        if self.config.console:
            print(f"Target  [{target.monitor or '?'}] {target.label}")
            print(f"Capture {backend.name}")
            print(f"Input   {input_backend.name}")
            if notes is not None:
                print(f"Notes   {notes.path} ({notes.char_count} chars loaded)")
            print(f"Output  {self.recorder.root}")

        self.messages: list[dict[str, Any]] = self._initial_messages()

    def _initial_messages(self) -> list[dict[str, Any]]:
        system_prompt = build_system_prompt()
        if self.input.name == "stub":
            system_prompt += (
                "\n\nInput is in stub/dry-run mode: actions are recorded but "
                "NOT sent to the game, so the board will not change."
            )
        if self.notes is not None:
            rendered_notes = self.notes.render_for_prompt()
            if rendered_notes:
                system_prompt += "\n\n" + rendered_notes

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        references = _reference_message()
        if references is not None:
            messages.append(references)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Start by calling view_board. Use one tool per turn."
                ),
            }
        )
        return messages

    def reset(self) -> None:
        """Reset the conversation and counters, starting a fresh run."""

        self.recorder.close()
        self.recorder = Recorder(
            self.config.out_dir,
            self.backend,
            self.input,
            getattr(self.agent, "model", "unknown"),
            getattr(self.agent, "prompt_cache_key", None),
            self.notes,
            self.target,
            self.config,
        )
        self.messages = self._initial_messages()
        self.frame = None
        self.last_frame_path = None
        self._prev_frame_png = None
        self.level_map = None
        self._move_history.clear()
        with self._guidance_lock:
            self._guidance.clear()
        self.action_count = 0
        self.step_index = 0
        self.final_text = None
        self._closed = False

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.event_callback is not None:
            self.event_callback({"type": event_type, **data})

    def add_guidance(self, text: str) -> bool:
        """Queue a user steering message for the next model turn."""
        text = text.strip()
        if not text:
            return False
        with self._guidance_lock:
            self._guidance.append(text)
        return True

    def _take_guidance(self) -> list[str]:
        with self._guidance_lock:
            items = list(self._guidance)
            self._guidance.clear()
        return items

    @property
    def pending_guidance(self) -> int:
        with self._guidance_lock:
            return len(self._guidance)

    def _capture(self) -> Frame:
        # Save the current frame as "previous" before capturing a new one,
        # so movement tools can compare before/after to detect blocked moves.
        if self.frame is not None:
            self._prev_frame_png = self.frame.png
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

    def _level_map_note(self) -> str:
        if not self.level_map:
            return ""
        return "Current level map:\n" + format_level_map(self.level_map)

    def _format_history(self) -> str:
        """Format recent move history as a compact summary."""
        if not self._move_history:
            return ""
        parts = []
        for entry in self._move_history[-5:]:
            tool = str(entry["tool"]).replace("move_", "")
            blocked = entry["blocked"]
            if blocked is True:
                parts.append(f"{tool}✗")
            elif blocked is False:
                parts.append(f"{tool}✓")
            else:
                parts.append(f"{tool}?")
        return f"Recent moves: {' '.join(parts)}."

    def _execute_tool(
        self, call: ModelToolCall
    ) -> tuple[ToolResult, Path | None]:
        try:
            if call.name == "level_map":
                updated = parse_level_map_args(
                    call.arguments,
                    self.level_map,
                )
                self.level_map = updated
                if not updated:
                    message = "Level map cleared."
                else:
                    height = len(updated)
                    width = len(updated[0])
                    message = (
                        f"Level map set ({height}x{width}).\n\n"
                        f"{format_level_map(updated)}"
                    )
                return ToolResult(ok=True, message=message), None

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

            if call.name == "reset_level":
                self.input.key("r")
                self.level_map = None
                self.action_count += 1
                self._pause()
                frame = self._capture()
                # After reset the level restarts, so don't compare against
                # the pre-reset frame for the next move.
                self._prev_frame_png = None
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Level reset with R; level map cleared. "
                            f"Screenshot attached "
                            f"({frame.width}x{frame.height})."
                        ),
                        image_png=frame.png,
                        state_changed=True,
                    ),
                    self.last_frame_path,
                )

            key = MOVE_KEYS.get(call.name)
            if key is not None:
                self.input.key(key)
                self.action_count += 1
                self._pause()
                frame = self._capture()
                # Detect whether the move actually changed the game state
                # by comparing the new screenshot with the previous one.
                blocked: bool | None = None
                if self._prev_frame_png is not None:
                    changed = _frames_differ(self._prev_frame_png, frame.png)
                    blocked = not changed

                # Record move in history (keep last 10)
                self._move_history.append({
                    "tool": call.name,
                    "blocked": blocked,
                })
                if len(self._move_history) > 10:
                    self._move_history = self._move_history[-10:]

                # Build history summary for the message
                history_str = self._format_history()

                if blocked:
                    message = (
                        f"Move {key} was BLOCKED — no step consumed. "
                        f"Do NOT repeat this move. Try a different direction "
                        f"or call view_board to re-assess. "
                        f"{history_str}"
                    )
                else:
                    message = (
                        f"Moved {key} one tile. Step consumed. "
                        f"{history_str} "
                        "Call view_board to see the result."
                    )
                return (
                    ToolResult(
                        ok=True,
                        message=message,
                        state_changed=not blocked if blocked is not None else True,
                        blocked=blocked,
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
        guidance = self._take_guidance()
        if guidance:
            text = "\n".join(f"- {item}" for item in guidance)
            for item in guidance:
                self.recorder.log_guidance(step, item)
            self.messages.append(
                {
                    "role": "user",
                    "content": "User guidance for this turn:\n" + text,
                }
            )
            self._emit("guidance", text=text, step=step)
            if self.config.console:
                print(f"  guidance: {text}")
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
        if result.image_png:
            result = replace(
                result,
                image_png=annotate_helltaker_grid(result.image_png),
            )
        map_updated = False
        if (
            result.ok
            and primary.name in MOVE_KEYS
            and result.blocked is False
        ):
            updated_map = apply_move_to_level_map(
                self.level_map,
                primary.name,
            )
            if updated_map is not None and updated_map != self.level_map:
                self.level_map = updated_map
                map_updated = True
        map_note = self._level_map_note()
        if result.ok and primary.name != "level_map" and map_note:
            result = replace(
                result,
                message=(
                    f"{result.message}\n\n{map_note}"
                    if result.message
                    else map_note
                ),
            )
        self.messages.append(
            result.as_tool_message(primary.id, primary.name)
        )
        self.recorder.log_tool_call(
            step, primary, result, next_frame, source_frame
        )
        if map_updated or (
            result.ok and primary.name in {"level_map", "reset_level"}
        ):
            self.recorder.log_level_map(step, self.level_map)
        self._emit(
            "tool_result",
            step=step,
            tool=primary.name,
            ok=result.ok,
            error=result.error,
            message=result.message,
            blocked=result.blocked,
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
                status = "BLOCKED" if result.blocked else "ok"
                print(f"  tool {primary.name}: [{status}] {result.message}")
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
            "Seconds to wait after a move before capturing (default: 0.5)."
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
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help=(
            "Mistral reasoning effort; defaults to MISTRAL_REASONING_EFFORT "
            "or the model default."
        ),
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
                reasoning_effort=args.reasoning_effort,
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
