"""Prompt templates exposed over MCP.

These mirror the guidance from the original harness system prompt, adapted for
an external agent loop: the client decides how to reason and when to speak,
while the server only supplies game rules and puzzle strategy.
"""

from __future__ import annotations

import re

RULES = """\
# Helltaker rules

You are controlling Helltaker, a grid-based puzzle game. The goal is to reach
the demon without running out of steps.

## Tools

- `view_board`: capture the current screen.
- `move(direction)`: press one arrow key. Directions are `up`, `down`, `left`,
  `right`. Each call moves exactly one tile and returns a fresh screenshot.
- `reset_level`: press R to restart the level.
- `toggle_grid(enabled)`: label every tile with its row and column
  (`r3c2`) so positions can be referenced precisely.
- `toggle_walkability(enabled)`: tint walkable tiles green and blocked tiles
  red.
- `set_boxes(boxes)` / `add_boxes(boxes)`: draw persistent labelled boxes on
  screenshots. Box coordinates are screenshot pixels:
  `{"box_2d": [xmin, ymin, xmax, ymax], "label": "demon"}`.

## Mechanics

- Skeleton knights can be kicked by moving into them. A skeleton kicked into
  the level edge is destroyed. Kicks do not cascade.
- Skeletons are not a threat by themselves, but they delay you and waste
  steps. Kicking every skeleton is usually a bad plan.
- Spikes are marked by white or black spots on the floor that alternate
  between hidden and visible as you move. They do not appear in early levels.
- Black tiles are edges and cannot be entered. The map is grid-based but not
  perfectly regular.
- Many obstacles are pushable by moving into them. Rocks can sit on red floor
  and still block movement.
- Assume only the red floor is walkable. Everything else is blocked.
- Moving into a wall or the level edge does not consume a step, but it wastes
  time. If a move is reported BLOCKED, change strategy instead of repeating
  it.
- Pillars are either obstacles or decoration. They are not interactive.
- The counter on the left is the remaining step count. `X` means the next
  move is your last.
- A white flash on a black screen means you ran out of steps and lost. After
  the reset, call `view_board` again to see the level.
"""

STRATEGY = """\
# Helltaker strategy

- **Study the board geometry first.** The route is not always a straight line.
  Look for corridors, gaps, and ways around obstacles. A wall of enemies does
  not always need to be fought through.
- **Distinguish obstacle types.** Some things block movement (walls, edges,
  solid objects), some can be pushed or kicked (skeletons, pushable blocks),
  and some are decoration.
- **Kicking is a tool, not a weapon.** A kick moves a skeleton one tile and
  costs the same step as a normal move. Use kicks to clear a path, push an
  enemy off an edge, or make space. Do not kick everything.
- **The demon is the goal.** Plan to reach the tile next to her. If the direct
  route is blocked, approach from another direction.
- **Count your steps.** Estimate the route length before committing. If it
  exceeds the remaining counter, find a shorter route.
- **Learn from blocked moves.** A BLOCKED result tells you that a wall, edge,
  or solid object is there. Update your mental map and probe a different
  direction.
- **Explore before committing.** When unsure, take one step, inspect the
  result, then continue.
- **Reset is a last resort.** Before resetting, ask what you learned and
  whether you can use it without restarting.
"""

WORKFLOW = """\
# Mandatory workflow

1. **Plan before moving.** After the first `view_board`, do not move yet.
   Identify:
   - where the demon is (the goal),
   - where Helltaker is (the character in the white suit),
   - where the skeletons, spikes, and obstacles are,
   - the shortest path to the demon,
   - how many steps that path needs versus how many remain.
   Only start moving once you have a complete plan.

2. **Mark the board.** Use `set_boxes` to label the demon, skeletons,
   Helltaker, and spike traps. Boxes persist on every future screenshot, so
   they preserve context even when older screenshots are pruned. Enable
   `toggle_grid` or `toggle_walkability` if coordinates or open paths are
   hard to read.

3. **Check every move result.** Each `move` reports whether the move was
   BLOCKED (no step consumed) or successful.
   - BLOCKED: do not repeat the direction. Try another direction or call
     `view_board` to re-assess.
   - Successful: continue along the plan.

4. **After a reset, change strategy.** If you call `reset_level`, do not
   replay the same sequence. Use what you learned and try a different route.
   The game can briefly show a title/loading screen after a reset; call
   `view_board` once more to see the level.

5. **Keep reasoning concise.** State the plan in a few sentences, then act.
   Do not re-analyse the same board state repeatedly.

6. **Record useful discoveries.** Use `append_notes` to save durable facts
   about a level (traps, required kicks, dead ends) so future runs can read
   them through `helltaker://notes` or `read_notes`.
"""

_HEADING_RE = re.compile(r"^#{1,6}\s")
_LEVEL_RE = re.compile(r"\blevel\s*#?\s*(\d+)\b", re.IGNORECASE)


def level_guide(level: int, notes_text: str = "") -> str:
    """Return known notes for a level, or general guidance.

    There is no built-in level database: the guide comes from the persistent
    notes file, which is populated by earlier runs.  Sections are matched by a
    Markdown heading containing "level N" (or by a line mentioning it).
    """

    sections = _split_sections(notes_text)
    matches = [
        section
        for section in sections
        if any(int(number) == level for number in _LEVEL_RE.findall(section))
    ]

    header = f"# Known notes for level {level}\n"
    if matches:
        return header + "\n\n".join(matches).strip() + "\n"

    return (
        header
        + "No level-specific notes have been recorded yet. Use the general "
        + "rules and strategy prompts, inspect the board with `view_board`, "
        + "and add useful facts with `append_notes` after playing.\n"
    )


def _split_sections(text: str) -> list[str]:
    if not text.strip():
        return []

    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.match(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(lines).strip() for lines in sections if lines]
