# itb-harness

Prototype LLM game harness. The first milestone is the **observe loop**: pick a
game window once per session, capture its client area, and write PNG frames plus
metadata for a model to consume.

## Current backends

- **KDE Plasma 6 / KWin Wayland** via [`kwcapture`](https://pypi.org/project/kwcapture/),
  which talks to `org.kde.KWin.ScreenShot2`. Per-window capture by KWin UUID.
- **Generic X11** via `python-xlib`. This works on Xorg, XWayland, Xvfb, and the
  nested Xwayland that **gamescope** creates. It captures a specific X11 window
  rather than the root, because the nested gamescope root is black.
- No structured game state is fed to the model; only frames.

## Install

```bash
uv sync --extra kde   # KDE/KWin backend
uv sync --extra x11   # generic X11 / gamescope backend
```

Both are optional extras because they are platform-specific. The rest of the
harness only depends on the backend interface in
`src/itb_harness/capture/base.py`.

## gamescope

Run the game in gamescope, then point the harness at gamescope's nested
Xwayland display:

```bash
gamescope -W 1920 -H 1080 -w 1920 -h 1080 -- ./Breach

# Find the game's nested display and auth file:
tr '\0' '\n' < /proc/<game-pid>/environ | grep -E '^(DISPLAY|XAUTHORITY)='

DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-harness --backend x11 --window Breach
```

The `x11` backend connects to that nested server directly. Input tools should
do the same (`DISPLAY=:2 pyautogui`); screenshots must capture the game window
or the gamescope window, not the nested X root. `auto` detects a non-`:0`
`DISPLAY` in a Wayland session as a nested X server, but `--backend x11` is
explicit and recommended.

## Run

```bash
uv run itb-harness
```

Useful commands inside the REPL:

```text
windows                 list windows
target 3                select by index
target Breach           select by title/app-id substring
observe                 capture one frame
auto 10 0.5             capture 10 frames, 0.5 s apart
status                  show target and output directory
quit
```

Non-interactive shortcuts:

```bash
uv run itb-harness --list
uv run itb-harness --window Breach --once --out runs
```

Output:

```text
runs/<timestamp>/
  run.json              backend, monitors, start time
  manifest.jsonl        one metadata record per captured frame
  frames/0001.png
```

The target window is held for the session; its backend identifier is
revalidated before each capture and the run fails if the window closes.

## Agent loop

The observer/act loop lives in `src/itb_harness/loop.py` and runs as
`itb-agent`.  Tool calling is not wired; the model replies with a JSON array of
point actions:

```json
[
  {"point_2d": [500, 250], "label": "click"},
  {"point_2d": [750, 250], "label": "hover"}
]
```

Coordinates are normalized to `0..1000`, origin top-left.  `click` performs a
left click; `hover` only moves the pointer.  Actions are executed in order, then
one new screenshot is captured.

```bash
export MISTRAL_API_KEY=...
export MISTRAL_MODEL=mistral-small-2603   # optional

DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-agent --backend x11 --window Breach --steps 10 --input x11
```

`--input stub` (the default) records clicks/hovers without touching the pointer.
`--input x11` performs real XTest input on `DISPLAY`; use it with the gamescope
nested display.  `--max-actions` caps a batch, and `--action-delay` inserts a
pause between inputs.

Each iteration:

1. capture the target window
2. send the screenshot to the model
3. parse the JSON point list
4. execute `click`/`hover` in order
5. capture a new screenshot and repeat until `--steps`

A malformed response is returned to the model as a text error without executing
anything or capturing a new frame.

Output:

```text
runs/<timestamp>-agent/
  run.json              backend, input backend, model, target, limits
  frames/0001.png       initial and post-batch screenshots
  actions.jsonl         point_2d, pixel/screen coords, label, result, frames
  conversation.jsonl    model responses
```

Old screenshots are pruned from the conversation (`max_images` in `LoopConfig`,
default 4) so the context does not grow without bound.

## Tool schemas (not wired)

`src/itb_harness/tools.py` is retained for a future structured-action mode, but
the current loop does **not** send or execute these tools.  Point actions are
the active interface.

| Tool | Args | Kind |
|---|---|---|
| `move_mech` | `mech` (1-3), `target` `{x,y}` | action |
| `prepare_attack` | `mech` (1-3), `weapon` (1-3) | action |
| `execute_attack` | `target` `{x,y}` | action |
| `end_turn` | — | action |
| `reset_turn` | — | action |
| `inspect` | `target` `{x,y}` | observation |

Board coordinates are 0-7, origin top-left.  `inspect` is intentionally
image-only: it will return a crop once the board grid is calibrated, not
structured game state.  Keep hidden state out of the model if the point is to
test vision and reasoning.

```python
from itb_harness.tools import ToolRegistry, StubActions

registry = ToolRegistry()                    # dry-run stubs
registry = ToolRegistry(StubActions(dry_run=False))  # "not implemented" errors
tools = registry.mistral_tools()             # pass as `tools=` to Mistral

result = registry.execute("move_mech", {"mech": 1, "target": {"x": 3, "y": 4}})
message = result.as_tool_message("call-123", "move_mech")
```

The stubs validate shapes/ranges and the prepare-then-fire sequence.  They
cannot validate board legality yet; that belongs in a `GameActions`
implementation backed by Lua or UI input.  A failed action should come back as
`ok=False` plus an error code, with `state_changed=False`; then the harness
should re-prompt with the same screenshot instead of capturing a new one.

## Cross-platform note

There is no single reliable Python API for **window** capture across Wayland,
macOS, and Windows. Screen capture is easy (`XGetImage`, `PrintWindow`,
ScreenCaptureKit) but window enumeration and native Wayland capture are not.
The design here is a small `CaptureBackend` protocol:

- `list_monitors()`
- `list_windows()`
- `capture_window(id) -> Frame`

Current implementations:

- KWin (KDE Wayland)
- Generic X11 via `python-xlib` (Xorg, XWayland, Xvfb, gamescope)

Future backends can be added without changing the REPL/agent loop:

- Linux Wayland (portable): xdg-desktop-portal ScreenCast (PipeWire).
- Windows: Windows Graphics Capture or `PrintWindow`.
- macOS: ScreenCaptureKit / `screencapture -l`.

Note: `mss` is not used. It only captures X root regions; the nested gamescope
Xwayland root is black, so the X11 backend uses `XGetImage` on the game window
instead.
