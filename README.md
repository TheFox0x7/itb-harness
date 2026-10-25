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

## Replay viewer

`itb-web` serves recorded runs from a `runs/` directory:

```bash
uv run itb-web --runs runs --port 8000
# open http://127.0.0.1:8000
```

The viewer reads `run.json`, `conversation.jsonl`, `actions.jsonl`, and
`frames/*.png` from each run. It provides:

- run selection
- per-turn thinking, model text, usage, and tool calls
- frame thumbnails and previous/next navigation
- the latest screenshot by default
- click/hover markers drawn from `click`/`inspect` points
- live pixel-coordinate readout and crosshair while hovering the image
- click the image to copy the current `x, y` to the clipboard
- marker toggle and keyboard left/right frame navigation

It is read-only and binds to `127.0.0.1` by default. It does not control a live
run yet.

The viewer implementation is in `src/itb_harness/web/server.py` with static
HTML/CSS/JS under `src/itb_harness/web/static/`.

## Interactive live UI

Start a live session with the browser UI:

```bash
DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-agent \
    --backend x11 --window Breach --input x11 \
    --web --port 8000
```

Open `http://127.0.0.1:8000`.

The session starts **paused**, so it will not run until you press **Step**.
Controls:

- **Step** — execute one model turn/tool call, then pause again
- **Resume** — run continuously until paused or stopped
- **Pause** — finish the current step, then pause
- **Stop** — stop after the current step
- **follow** — keep the browser on the live run and latest frame

There is no predetermined step count in web mode; `--steps` is ignored. The
live UI reloads tool calls, frames, and timeline data roughly every 600 ms.
Harness console output is suppressed in web mode; thinking appears in the
browser after each turn completes. Token-level thinking is still only streamed
in the terminal when running headless.

The live server also serves the replay UI for all runs under `--out` (default
`runs/`). `itb-web` remains a read-only replay server for recorded runs.

## Agent loop

The observer/act loop lives in `src/itb_harness/loop.py` and runs as
`itb-agent`.

The model is not given an automatic screenshot. It has the following tools,
plus an optional persistent-note tool:

| Tool | Args | Effect |
|---|---|---|
| `view_board` | — | Capture and return the current game screenshot |
| `click` | `point_2d` pixels or normalized | Move, left-click, and return a screenshot marked at the click |
| `inspect` | `point_2d`, optional `with_details` | Hover and capture; `with_details` holds Left Ctrl while capturing |
| `attack_order` | — | Hold Left Alt and capture the attack-order preview |
| `repair` | — | Press `R`; then click the selected mech as the repair target |
| `weapon` | `number` 1 or 2 | Press that number; then click the target |
| `select_unit` | `number` 1-5 | Press A/S/D/Z/X for units 1-5 |
| `end_turn` | — | Press Space |
| `undo_move` | — | Press Left Shift; only moves can be undone, attacks are permanent |
| `reset_turn` | — | Press Backspace; once per mission and not always available |
| `cancel` | — | Right-click |
| `note` | `text` | Append a note for future runs (only with `--notes`) |

Coordinates default to `[x, y]` in screenshot pixels, origin top-left; the
valid range is `0..width-1` by `0..height-1`. `view_board` reports the
screenshot dimensions. Pass `--normalize-coordinates` to switch click/inspect
points to normalized `0..1000` values instead; use `--no-normalize-coordinates`
to return to pixels. `repair`, `weapon`, and `select_unit` operate on the
current UI selection; select mechs and weapons with clicks or the unit keys.

`click`, `inspect`, and `attack_order` return screenshots. `click` annotates
the returned frame at the clicked pixel. `repair`, `weapon`, `select_unit`,
`end_turn`, `undo_move`, `reset_turn`, and `cancel` return text only; call
`view_board` when you need to see the result.

The model is instructed to call `view_board` first and to call it again after
an action when it needs to see the result. `inspect` is for tooltips/hover
states and returns the screenshot captured after hovering.

```bash
export MISTRAL_API_KEY=...
export MISTRAL_MODEL=mistral-large-4   # optional

DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-agent --backend x11 --window Breach --steps 10 --input x11
```

`--input stub` (the default) records clicks/hovers without touching the pointer.
`--input x11` performs real XTest input on `DISPLAY`; use it with the gamescope
nested display. `--action-delay` waits after clicks, keys, and hover actions;
`--max-images` controls how many tool screenshots stay in history (`0` keeps
everything). `--normalize-coordinates` selects normalized 0..1000 points for
click/inspect; the default is screenshot pixels.

`--snap-to-tile` snaps click/inspect coordinates to the nearest board tile
centre. The isometric grid is calibrated at runtime from the tan drop-zone
tiles during deployment (colour masking + connected components), then reused
for the rest of the game. Clicks farther than 60 px from any tile centre are
left unchanged (assumed to target UI elements). The snap is logged in the
action record and mentioned in the tool result message.

`--grid-overlay` draws the isometric tile grid on every screenshot: diamond
outlines for all 64 tiles, centre markers, and pixel-coordinate labels
(e.g. `472,288`) at each tile centre. During deployment, valid drop-zone
tiles are highlighted with a gold tint. The overlay makes it easy for the
model to read off exact click coordinates instead of estimating them.
Implies grid calibration.

Combine both flags for the best results:

```bash
uv run itb-agent --backend x11 --window Breach --input x11 \
  --snap-to-tile --grid-overlay --steps 30
```

Action logs always include the resolved `pixel` coordinate, so replay/live
markers work in either coordinate mode even when the model’s arguments are
normalized.

Enable persistent notes with:

```bash
uv run itb-agent ... --notes --notes-file notes.md
```

Existing notes are loaded into the system prompt at startup; the optional
`note` tool appends new entries for future runs. Notes are not injected
mid-run, which keeps the prompt prefix stable. `--notes-max-chars` limits how
much of a long notes file is loaded.

Each iteration:

1. send the tool schemas and conversation to the model
2. execute the single tool call it returns
3. append the tool result (text or image)
4. repeat until `--steps` or the model returns no tool call

Thinking is streamed live by default (`--no-stream` disables it). It is also
always saved to disk when the model provides it. With `--no-stream`, use
`--show-thinking` to print the saved reasoning after the response.

Output:

```text
runs/<timestamp>-agent/
  run.json              backend, input, model, cache key, limits
  frames/0001.png       screenshots returned by view_board/inspect
  thinking/0001.txt     model reasoning/thinking chunks, if provided
  actions.jsonl         tool name, args, result, frame links
  conversation.jsonl    model content, thinking, usage, tool calls
```

Old tool-message screenshots are pruned from the conversation by default
(`--max-images 4`) so the context stays bounded. Pruning changes the historical
prompt prefix and invalidates cache entries from that point onward; use
`--max-images 0` for append-only history and better prompt-cache reuse.

Notes live in a separate Markdown file (`notes.md` by default), outside the run
directory, so they persist across runs and are easy to review.

`Ctrl-C` stops the loop after the current tool call; a fully interruptible,
open-ended run can replace the `--steps` cap later without changing the tool
protocol.

### Prompt caching

The agent sends a stable `prompt_cache_key` on every request (per-run UUID,
`--cache-key`, or `MISTRAL_PROMPT_CACHE_KEY`). Responses report cached prompt
tokens in `usage`, which are logged and shown as:

```text
  cache: 64/1024 prompt tokens
```

Mistral caching is prefix-based and best-effort.

## Legacy tool schemas (not used)

`src/itb_harness/tools.py` still contains the earlier semantic action tools
(`move_mech`, `end_turn`, etc.) and `ToolResult`, but the loop does not send
those schemas. The active board tools are defined in
`src/itb_harness/board_tools.py`.

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
