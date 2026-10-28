---
name: helltaker
description: Play the Helltaker puzzle game through the itb_harness MCP server.
user-invocable: true
---

# Play Helltaker

Use the `itb_harness` MCP server. Vibe exposes its tools with the server
prefix, for example `itb_harness_view_board`, `itb_harness_move`, and
`itb_harness_start_run`.

## Start

1. Call `itb_harness_list_windows` if the target window is unclear, then
   `itb_harness_select_window` with the Helltaker window.
2. If your client exposes MCP prompts, read `helltaker-rules` and
   `helltaker-strategy`; otherwise rely on this skill and the server's
   `view_board` screenshots.
3. Call `itb_harness_start_run` before a serious attempt so frames and actions
   are recorded.
4. Call `itb_harness_view_board` to see the board.

## Plan, then act

- Find the demon (goal), Helltaker (you, white suit), skeletons, spikes, and
  obstacles.
- Enable `itb_harness_toggle_grid` when you need to reference tiles as
  `r3c2`.
- Enable `itb_harness_toggle_walkability` to distinguish red walkable floor
  from black edges and solid obstacles.
- Mark key entities with `itb_harness_set_boxes`:
  `{"boxes": [{"box_2d": [xmin, ymin, xmax, ymax], "label": "demon"}]}`.
  Boxes persist on every later screenshot.
- Count the route length against the remaining step counter before committing.

## Move

- Call `itb_harness_move` with one of `up`, `down`, `left`, `right`.
- Each successful move returns a fresh screenshot and says a step was
  consumed.
- A `BLOCKED` result means nothing changed. Do not repeat that direction:
  choose another one or call `itb_harness_view_board` to re-assess.
- Kicking a skeleton is a normal move; use it to clear a path or push a
  skeleton off an edge, not to fight every enemy.
- After a mistake, learn from the blocked moves and try a different route
  before calling `itb_harness_reset_level`. After a reset, change strategy
  rather than replaying the same sequence.

## Finish

- When the level is complete (or you stop), call `itb_harness_end_run`.
- Save durable level knowledge with `itb_harness_append_notes`, for example
  trap timing, required kicks, or dead ends.
- Review past attempts through `helltaker://runs` and
  `helltaker://runs/{run_id}/frames/{name}`.

Keep reasoning short: inspect, plan, move, verify. Do not describe an action
in prose when you can call the tool.
