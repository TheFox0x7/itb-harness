"""Validate a proposed Helltaker solution against the model-authored map."""

from __future__ import annotations

from dataclasses import dataclass


MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (-1, 0),
    "move_down": (1, 0),
    "move_left": (0, -1),
    "move_right": (0, 1),
}

_HELLTAKER = "H"
_SKELETON = "S"
_ROCK = "R"
_SPIKE = "T"
_WALL = "X"
_DEMON = "D"
_EMPTY = "."
_KEY = "K"
_BARRIER = "B"

_SPIKE_MODES = {"static", "up", "down"}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    reason: str
    steps: int
    cost: int
    player: tuple[int, int] | None = None
    demon: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class _State:
    player: tuple[int, int]
    rocks: frozenset[tuple[int, int]]
    skeletons: frozenset[tuple[int, int]]
    spikes: frozenset[tuple[int, int]]
    keys: frozenset[tuple[int, int]]
    barriers: frozenset[tuple[int, int]]


def validate_solution(
    rows: list[list[str]],
    moves: list[str],
    max_steps: int,
    spike_mode: str,
) -> ValidationResult:
    """Replay a proposed move list against the map.

    Success requires every move to be legal and H to finish adjacent to D.
    Spikes are walkable. Hidden spikes cost 2 will; alternating extended
    spikes cost 1 because they retract before the player enters. Static
    spikes are treated as extended and cost 2. Rocks and skeletons may cover
    spikes.
    """

    if spike_mode not in _SPIKE_MODES:
        raise ValueError(
            "spike_mode must be one of: static, up, down."
        )
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    height, width, walls, start, demon = _parse_map(rows)
    state = start
    spikes_up = spike_mode != "down"
    cost = 0

    for index, move in enumerate(moves, start=1):
        delta = MOVE_DELTAS.get(move)
        if delta is None:
            return ValidationResult(
                ok=False,
                reason=f"unknown move {move!r} at step {index}",
                steps=index - 1,
                cost=cost,
                player=state.player,
                demon=demon,
            )

        target = (
            state.player[0] + delta[0],
            state.player[1] + delta[1],
        )
        if target in state.spikes:
            # Hidden spikes cost an extra will. An alternating extended spike
            # retracts before entry, so it only costs the normal move.
            move_cost = 1 if spikes_up and spike_mode != "static" else 2
        else:
            move_cost = 1

        outcome = _apply_move(
            height,
            width,
            walls,
            demon,
            state,
            delta,
        )
        if isinstance(outcome, str):
            return ValidationResult(
                ok=False,
                reason=f"step {index} ({move}): {outcome}",
                steps=index - 1,
                cost=cost,
                player=state.player,
                demon=demon,
            )

        cost += move_cost
        if cost > max_steps:
            return ValidationResult(
                ok=False,
                reason=(
                    f"step {index} ({move}) reaches will cost {cost} but "
                    f"max_steps is {max_steps}"
                ),
                steps=index,
                cost=cost,
                player=outcome.player,
                demon=demon,
            )

        state = outcome
        if spike_mode != "static":
            spikes_up = not spikes_up

    if _adjacent(state.player, demon):
        return ValidationResult(
            ok=True,
            reason="solution reaches a tile adjacent to the demon",
            steps=len(moves),
            cost=cost,
            player=state.player,
            demon=demon,
        )
    return ValidationResult(
        ok=False,
        reason=(
            "final player position "
            f"{_position(state.player)} is not adjacent to the demon "
            f"{_position(demon)}"
        ),
        steps=len(moves),
        cost=cost,
        player=state.player,
        demon=demon,
    )


def _parse_map(
    rows: list[list[str]],
) -> tuple[
    int,
    int,
    frozenset[tuple[int, int]],
    _State,
    tuple[int, int],
]:
    if not rows or not rows[0]:
        raise ValueError("Level map is empty.")

    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                "All level-map rows must have the same width; "
                f"row 0 has {width} cells but row {index} has {len(row)}."
            )

    player: tuple[int, int] | None = None
    demon: tuple[int, int] | None = None
    rocks: set[tuple[int, int]] = set()
    skeletons: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    keys: set[tuple[int, int]] = set()
    barriers: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()

    for row, cells in enumerate(rows):
        for col, value in enumerate(cells):
            symbol = value.upper()
            position = (row, col)
            if symbol == _HELLTAKER:
                if player is not None:
                    raise ValueError("Level map must contain exactly one H.")
                player = position
            elif symbol == _DEMON:
                if demon is not None:
                    raise ValueError("Level map must contain exactly one D.")
                demon = position
            elif symbol == _ROCK:
                rocks.add(position)
            elif symbol == _SKELETON:
                skeletons.add(position)
            elif symbol == _SPIKE:
                spikes.add(position)
            elif symbol == _KEY:
                keys.add(position)
            elif symbol == _BARRIER:
                barriers.add(position)
            elif symbol == _EMPTY:
                continue
            elif symbol == "?":
                raise ValueError(
                    "Level map contains '?'; use only observed cells."
                )
            elif symbol == _WALL:
                walls.add(position)
            else:
                raise ValueError(
                    f"Unsupported level-map symbol {value!r} at "
                    f"{_position(position)}."
                )

    if player is None:
        raise ValueError("Level map must contain H.")
    if demon is None:
        raise ValueError("Level map must contain D.")

    return (
        len(rows),
        width,
        frozenset(walls),
        _State(
            player=player,
            rocks=frozenset(rocks),
            skeletons=frozenset(skeletons),
            spikes=frozenset(spikes),
            keys=frozenset(keys),
            barriers=frozenset(barriers),
        ),
        demon,
    )


def _apply_move(
    height: int,
    width: int,
    walls: frozenset[tuple[int, int]],
    demon: tuple[int, int],
    state: _State,
    delta: tuple[int, int],
) -> _State | str:
    target = (
        state.player[0] + delta[0],
        state.player[1] + delta[1],
    )
    if not _inside(target, height, width):
        return "destination is outside the map"
    if target == demon:
        return "cannot move onto the demon; finish adjacent to it"

    if target in state.barriers:
        return "destination is a barrier"
    if target in state.rocks:
        return _push_rock(
            height,
            width,
            walls,
            demon,
            state,
            target,
            delta,
        )
    if target in state.skeletons:
        return _kick_skeleton(
            height,
            width,
            walls,
            demon,
            state,
            target,
            delta,
        )
    if target in walls:
        return "destination is a wall"
    if target in state.keys:
        return _State(
            player=target,
            rocks=state.rocks,
            skeletons=state.skeletons,
            spikes=state.spikes,
            keys=state.keys - {target},
            barriers=frozenset(),
        )
    if target in state.spikes:
        # Spikes cost extra will but remain in place; their state only
        # affects the cost of entering them.
        return _State(
            player=target,
            rocks=state.rocks,
            skeletons=state.skeletons,
            spikes=state.spikes,
            keys=state.keys,
            barriers=state.barriers,
        )

    return _State(
        player=target,
        rocks=state.rocks,
        skeletons=state.skeletons,
        spikes=state.spikes,
        keys=state.keys,
        barriers=state.barriers,
    )


def _push_rock(
    height: int,
    width: int,
    walls: frozenset[tuple[int, int]],
    demon: tuple[int, int],
    state: _State,
    rock: tuple[int, int],
    delta: tuple[int, int],
) -> _State | str:
    destination = (rock[0] + delta[0], rock[1] + delta[1])
    if not _inside(destination, height, width):
        return "rock would be pushed outside the map"
    if destination == demon:
        return "rock would be pushed onto the demon"
    if destination in walls:
        return "rock would be pushed into a wall"
    if destination in state.barriers:
        return "rock would be pushed into a barrier"
    if destination in state.keys:
        return "rock would be pushed onto a key"
    if destination in state.rocks or destination in state.skeletons:
        return "rock is blocked"

    rocks = (state.rocks - {rock}) | {destination}
    return _State(
        player=rock,
        rocks=frozenset(rocks),
        skeletons=state.skeletons,
        spikes=state.spikes - {destination},
        keys=state.keys,
        barriers=state.barriers,
    )


def _kick_skeleton(
    height: int,
    width: int,
    walls: frozenset[tuple[int, int]],
    demon: tuple[int, int],
    state: _State,
    skeleton: tuple[int, int],
    delta: tuple[int, int],
) -> _State:
    destination = (
        skeleton[0] + delta[0],
        skeleton[1] + delta[1],
    )
    skeletons = state.skeletons - {skeleton}
    spikes = state.spikes

    if _inside(destination, height, width):
        if (
            destination != demon
            and destination not in walls
            and destination not in state.rocks
            and destination not in state.barriers
            and destination not in state.keys
        ):
            skeletons = skeletons | {destination}
            spikes = spikes - {destination}

    return _State(
        player=skeleton,
        rocks=state.rocks,
        skeletons=frozenset(skeletons),
        spikes=frozenset(spikes),
        keys=state.keys,
        barriers=state.barriers,
    )


def _inside(
    position: tuple[int, int], height: int, width: int
) -> bool:
    row, col = position
    return 0 <= row < height and 0 <= col < width


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _position(position: tuple[int, int]) -> str:
    return f"r{position[0]}c{position[1]}"
