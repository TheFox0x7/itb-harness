# itb-harness

Prototype LLM game harness for **Helltaker**. Pick the game window once per
session, capture its client area, and let the model observe screenshots and
move with arrow-key tools.

## Branches

Every branch below forks from `main` (at `508c901`) and represents a different
approach shift that might be of interest:

| Branch | Approach |
|---|---|
| `main` | Base harness: window capture (KDE/KWin Wayland and generic X11), the replay viewer, the live web UI, and the tool-based agent loop documented below. |
| `helltaker` | Helltaker-specific iteration: game-change tests, steering and helper tools (frame annotation, board overlay), and capture/session utilities. |
| `helltaker-mcp` | `helltaker` plus swapping the in-process tool loop for an MCP server (`src/itb_mcp/`) exposing game state, frame diffs, overlays, prompts, and run recording, along with a Vibe skill (`vibe/skills/helltaker/`) that plays through the server. |
| `responses` | `helltaker` plus moving the agent adapter to Mistral's Responses API (`client.responses.create`), dropping bounding-box click tools in favor of a grid overlay and a model-maintained `level_map`. |
| `planner` | `responses` plus a plan-then-execute approach: `validate_solution` / `execute_plan` replay a proposed move list against the level map (`src/itb_harness/validator.py`) before touching the game, and reference crops of terrain and items are sent with the first turn. |

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

## Taskfile

A [Taskfile.yml](Taskfile.yml) wraps every mode so you don't have to dig
through this README. Run `task --list` (binary: `task` or `go-task`) to see
all of them:

```bash
task capture            # interactive capture REPL (itb-harness)
task capture:list       # list capturable windows
task capture:once       # one frame of WINDOW=Helltaker into runs/
task agent              # headless agent loop (itb-agent)
task live               # agent with the browser UI (itb-agent --web)
task replay             # read-only replay viewer (itb-web)
task game               # Helltaker under gamescope
task sync EXTRA=kde     # install deps (EXTRA=kde or x11)
task test               # pytest
task lint               # ruff check
```

Variables override the defaults, extra flags pass through with `CLI_ARGS`:

```bash
task live BACKEND=x11 WINDOW=Helltaker INPUT=x11 PORT=9000
task agent STEPS=30 CLI_ARGS="--max-images 0"
```

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
- **Reset** — clear the model conversation and counters, then start a fresh run
  directory without restarting the server
- **follow** — keep the browser on the live run and latest frame
- **steer** — queue a guidance message for the model's next turn

There is no predetermined step count in web mode; `--steps` is ignored.

The frame toolbar has a **grid** toggle. It draws a viewer overlay with 68×72
px cells: nine columns centered horizontally. The bottom of the grid is anchored
to the first row, scanning upward from the bottom of the image, that contains at
least one cell-width of continuous red floor pixels (`#8c1c31` / `#700c29`).
Rows propagate upward from that anchor, so the top row is partial when the
anchored height is not a multiple of 72 px. The web overlay also draws the
current `level_map` symbols into the matching grid cells, centered horizontally
and bottom-aligned. Each cell is labeled `r{row}c{col}` with row 0 at the
bottom of the grid. Grid lines are always drawn onto screenshots returned to the model;
map-symbol annotations are optional there and enabled with `toggle_map_overlay`.
`reset_level` captures a fresh frame and recomputes the anchor and overlay.
Saved raw frames remain unmodified.

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
| `reset_level` | — | Press `R` to restart the current level and clear the level map |
| `level_map` | `rows`, `updates`, or `clear` | Store the model's own text grid of the level |
| `validate_solution` | `moves`, `max_steps`, `spike_mode` | Replay a proposed move list against the current map |
| `execute_plan` | `moves`, `max_steps`, `spike_mode` | Validate and then execute the plan on the game |
| `press_space` | — | Press spacebar for dialogue/confirmations; never part of a plan |
| `toggle_map_overlay` | `enabled` bool | Show or hide the model's map symbols on its screenshots |

Movement tools capture a fresh frame for blocked-move detection and replay, but
do not return a screenshot to the model. Call `view_board` when you need to see
the result.

`level_map` is a model-maintained text scratchpad. The model can set a full
grid and later update individual cells:

```json
{"rows": ["[X] [X] [X] [X]", "[X] [H] [S] [D]", "[X] [X] [X] [X]"]}
{"updates": [{"row": 1, "col": 1, "symbol": "."}]}
{"clear": true}
```

The grid is stored by the harness, returned in the tool result, and appended to
later tool results as `Current level map:` so the model can keep it in sync
with the game. It is also shown in the web UI sidebar for both live runs and
replays, and its symbols are drawn onto the web grid overlay. On successful
moves the harness updates the map automatically: `H` advances, skeletons are
kicked one cell and destroyed against walls/rocks, and rocks are pushed only
into empty or spike cells. Blocked moves leave it unchanged. Use
`toggle_map_overlay` to draw the map symbols onto model screenshots for visual
row validation. `reset_level` clears it.

`validate_solution` replays a proposed move list against the current map. It
does not move the character. Validation succeeds only when every move is legal,
the total will cost is within `max_steps`, and `H` finishes on a tile adjacent
to `D`. Moving onto `D` itself, walls, out-of-bounds cells, or an immovable
rock fails validation. Rocks can be pushed into empty cells or spikes, and
skeletons are kicked one cell forward or destroyed against walls/rocks.
Stepping on a key (`K`) removes it and every barrier (`B`); moving into a
barrier while any remain fails validation like a wall.

`spike_mode` is `static` (spikes stay extended), `up` (start extended and
alternate each move), or `down` (start hidden and alternate each move). Hidden
spikes cost 2 will total. An alternating spike that is currently extended
costs only the normal move, because it retracts before the player enters the
tile. Static extended spikes cost 2 will. The validator rejects maps containing
`?`, since every tile should be observed.

`execute_plan` revalidates the same arguments, then plays the moves one at a
time. It aborts on a blocked move or if the level map cannot be updated. Keys
and barriers update the map during execution. `press_space` is a separate
dialogue/confirm action; only `move_*` entries are accepted in plans. Pressing
space outside dialogue does nothing. After reaching the demon, one dialogue
choice is correct and the other restarts the level; if the level restarts,
replay the same winning sequence and choose the other option.

The model is told that a level can extend beyond the visible 9-column grid or
the current screen, so its map may include off-screen rows and columns. One
level places the demon beyond the initial screen/grid, behind a desk; there,
reaching the tile in front of the desk is the win. For that level the model can
mark the desk cell as `D`, making the validator’s “finish adjacent to D” rule
match the actual win condition.

The prompt also tells the model that black or very dark filled squares are
solid and impassable, and that dark red rocks are impassable rather than
pushable. Both should be mapped as `X`; `R` is reserved for lighter, pushable
rocks.

At the start of the conversation the model also receives reference crops from
`src/itb_harness/references/` for extended spikes, hidden spikes, passable
floor tiles, pushable rocks, Helltaker, impassable rocks, and impassable walls,
so it can distinguish entities and hazards visually.

The model is instructed to call `view_board` first, plan the whole level
before moving, and call it again after movement when it needs to re-orient
itself. The counter on the left is the step counter; `reset_level` restarts the
level with `R` when a plan fails.

```bash
export MISTRAL_API_KEY=...
export MISTRAL_MODEL=mistral-large-4   # optional
export MISTRAL_REASONING_EFFORT=high   # optional

DISPLAY=:2 XAUTHORITY=/run/user/1000/xauth_... \
  uv run itb-agent --backend x11 --window Helltaker --steps 30 --input x11
```

`--reasoning-effort` overrides `MISTRAL_REASONING_EFFORT`; when neither is
set, the model's default reasoning behavior is used.

`--input stub` (the default) records key presses without sending them. Use
`--input x11` for real XTest input on `DISPLAY`.

`--action-delay` waits after a move before capturing (default `0.5`). Increase
it if the game’s movement animation is not finished before the next screenshot.
`--move-verify-attempts` and `--move-verify-delay` control extra captures used
before declaring a move blocked (defaults: `3` attempts, `0.15` seconds).

`--max-images` controls how many tool screenshots stay in conversation history
(`0` keeps everything).

The adapter in `src/itb_harness/agent.py` uses Mistral's Responses API
(`client.responses.create`) with the full conversation sent as `input` on every
turn. Function calls and tool results are exchanged as `function_call` and
`function_call_output` items.

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
