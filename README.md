# itb-harness

Prototype LLM game harness for **Helltaker**. Pick the game window once per
session, capture its client area, and let the model observe screenshots and
move with arrow-key tools.

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
gamescope -W 1920 -H 1080 -w 1920 -h 1080 -- ./Helltaker

# Find the game's nested display and auth file:
tr '\0' '\n' < /proc/<game-pid>/environ | grep -E '^(DISPLAY|XAUTHORITY)='

DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-harness --backend x11 --window Helltaker
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
target Helltaker           select by title/app-id substring
observe                 capture one frame
auto 10 0.5             capture 10 frames, 0.5 s apart
status                  show target and output directory
quit
```

Non-interactive shortcuts:

```bash
uv run itb-harness --list
uv run itb-harness --window Helltaker --once --out runs
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
- markers for older click/hover/inspect runs
- live coordinate readout and crosshair while hovering the image
- optional normalized `0–1000` coordinate readout/copy mode
- click the image to copy the current coordinates to the clipboard
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
    --backend x11 --window Helltaker --input x11 \
    --web --port 8000
```

Open `http://127.0.0.1:8000`.

The session starts **paused**, so it will not run until you press **Step**.
Controls:

- **Step** — execute one model turn/tool call, then pause again
- **Resume** — run continuously until paused or stopped
- **Pause** — finish the current step, then pause
- **Stop** — stop after the current step (an in-flight model request may finish)
- **follow** — keep the browser on the live run and latest frame
- **steer** — queue a guidance message for the model's next turn

There is no predetermined step count in web mode; `--steps` is ignored.

Steering is cooperative: the message is added to the model conversation at
the start of the next turn, not sent as an immediate key press. If the session
is paused, press **Step** or **Resume** after queueing guidance. The live bar
shows how many guidance messages are pending, and the guidance history shows
when a message is queued and when it is sent to the model.

The live UI uses an SSE stream at `/api/live/events` for:

- `thinking` and `content` deltas
- phase/status changes
- tool results and frame notifications
- queued/sent steering guidance
- first-token latency and last-activity age

The thinking panel streams live, and the status line shows phases such as
`model_waiting`, `thinking`, `content`, `tool`, and `paused`. A warning appears
after 15 seconds without model activity and a stalled style after 30 seconds.
After **Stop**, the UI closes the SSE stream and ignores late thinking/content
events instead of replaying them. Run files are still polled as a fallback for
timeline/frame refresh.

The timeline opens at the latest turn and preserves its scroll position and
expanded thinking blocks across live refreshes. If you are at the bottom, it
follows new turns; otherwise it stays where you were reading. Thinking text is
loaded into the DOM only when its block is expanded, so long runs stay light.
Steering interjections are shown in the timeline and recorded in
`guidance.jsonl`.

Harness console output is suppressed in web mode. Token-level thinking is
streamed only in the browser in web mode, and in the terminal in headless mode.
`--model-timeout` sets the Mistral request timeout (default 180 seconds); use
**Stop** if a request appears stuck for longer than that.

The live server also serves the replay UI for all runs under `--out` (default
`runs/`). `itb-web` remains a read-only replay server for recorded runs.

## Agent loop

The observation/movement loop lives in `src/itb_harness/loop.py` and runs as
`itb-agent`.

The model is not given an automatic screenshot. It has exactly these tools:

| Tool | Args | Effect |
|---|---|---|
| `view_board` | — | Capture and return the current game screenshot |
| `move_up` | — | Press Up and move one tile |
| `move_down` | — | Press Down and move one tile |
| `move_left` | — | Press Left and move one tile |
| `move_right` | — | Press Right and move one tile |
| `reset_level` | — | Press `R` to restart the current level and clear the box overlay |
| `set_boxes` | `boxes` list | Replace the persistent bounding-box overlay on screenshots |
| `add_boxes` | `boxes` list | Append boxes to the persistent overlay without clearing it |

Each movement tool captures a fresh screenshot after moving and returns it to
the model.

`set_boxes` takes a list of objects:

```json
{"boxes": [{"box_2d": [xmin, ymin, xmax, ymax], "label": "rock"}]}
```

Coordinates are screenshot pixels by default. Pass `--scaled-boxes` to use
scaled coordinates from 1 to 1000 instead, where 1 is the top/left edge and
1000 is the bottom/right edge.

The boxes are drawn on the current screenshot and every screenshot after it.
A later `set_boxes` call replaces the overlay completely; an empty list clears
it. `add_boxes` appends to the current overlay instead. `reset_level` also
clears the overlay. Known labels get category colors (`spike trap`,
`skeleton`, `rock`, `demon`, `helltaker`).

Use `--strict-box-labels` to restrict labels to those five categories and
enable the strict tool schema. Without the flag, arbitrary labels are allowed.

The model is instructed to call `view_board` first, plan the whole level
before moving, and call it again after movement when it needs to re-orient
itself. The counter on the left is the step counter; `reset_level` restarts the
level with `R` when a plan fails.

```bash
export MISTRAL_API_KEY=...
export MISTRAL_MODEL=mistral-large-4   # optional

DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-agent --backend x11 --window Helltaker --steps 30 --input x11
```

`--input stub` (the default) records key presses without sending them. Use
`--input x11` for real XTest input on `DISPLAY`.

`--action-delay` waits after a move before capturing (default `0.5`). Increase
it if the game’s movement animation is not finished before the next screenshot.

`--max-images` controls how many tool screenshots stay in conversation history
(`0` keeps everything).

`--strict-box-labels` restricts `set_boxes`/`add_boxes` labels to the known
categories: `spike trap`, `skeleton`, `rock`, `demon`, and `helltaker`. Boxes
are colored by category either way; the flag controls validation and the tool
schema. `--scaled-boxes` switches box coordinates from screenshot pixels to
scaled 1..1000 values. The timeline prints the box arguments, including
invalid calls, rather than only showing a box count.

Each iteration:

1. send the tool schemas and conversation to the model
2. execute the single tool call it returns
3. append the tool result, including a screenshot for movement and `view_board`
4. repeat until `--steps` or the model returns no tool call

Thinking is streamed live by default (`--no-stream` disables it). It is also
always saved to disk when the model provides it. With `--no-stream`, use
`--show-thinking` to print the saved reasoning after the response.

Output:

```text
runs/<timestamp>-agent/
  run.json              backend, input, model, cache key, limits
  frames/0001.png       screenshots returned by view_board and movement tools
  thinking/0001.txt     model reasoning/thinking chunks, if provided
  actions.jsonl         tool name, args, result, frame links
  conversation.jsonl    model content, thinking, usage, tool calls
  guidance.jsonl        user steering messages, by turn
```

Old tool-message screenshots are pruned from the conversation by default
(`--max-images 4`) so the context stays bounded. Pruning changes the historical
prompt prefix and invalidates cache entries from that point onward; use
`--max-images 0` for append-only history and better prompt-cache reuse.

`Ctrl-C` stops the loop after the current tool call.
