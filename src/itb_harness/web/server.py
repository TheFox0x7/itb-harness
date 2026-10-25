"""Read-only replay viewer for recorded harness runs.

The loop already writes ``run.json``, ``conversation.jsonl``, ``actions.jsonl``,
and ``frames/*.png``.  This module reconstructs a timeline from those files and
serves a small browser UI.  It does not control a live run.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


_STATIC_DIR = Path(__file__).parent / "static"
_FRAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")
_RUN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _basename(value: Any) -> str | None:
    if not value:
        return None
    return Path(str(value)).name


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    point = value.get("point_2d")
    if not isinstance(point, list) or len(point) != 2:
        return None
    if not all(isinstance(item, (int, float)) for item in point):
        return None
    return [float(point[0]), float(point[1])]


def _action_from_record(record: dict[str, Any]) -> dict[str, Any]:
    action = dict(record)
    action["step"] = int(action.get("step", 0))
    pixel = action.get("pixel")
    if (
        isinstance(pixel, list)
        and len(pixel) == 2
        and all(isinstance(item, (int, float)) for item in pixel)
    ):
        action["point_2d"] = [float(pixel[0]), float(pixel[1])]
    else:
        action["point_2d"] = _point(action.get("arguments"))
    action["frame_name"] = _basename(action.get("frame"))
    action["source_frame_name"] = _basename(action.get("source_frame"))

    marker_frames: list[str] = []
    if action.get("ok", True) and action["point_2d"] is not None:
        if action.get("tool") == "click" and action["source_frame_name"]:
            marker_frames.append(action["source_frame_name"])
        elif action.get("tool") in {"hover", "inspect"}:
            for name in (
                action["source_frame_name"],
                action["frame_name"],
            ):
                if name and name not in marker_frames:
                    marker_frames.append(name)
    action["marker_frames"] = marker_frames

    args = action.get("arguments")
    if isinstance(args, dict) and "with_details" in args:
        # Preserve the old inspect flag for runs recorded before the split.
        action["with_details"] = bool(args.get("with_details"))
    else:
        action["with_details"] = action.get("tool") == "inspect"
    return action


def _build_turns(
    conversation: list[dict[str, Any]], actions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    conversations: dict[int, dict[str, Any]] = {}
    for record in conversation:
        try:
            step = int(record.get("step", 0))
        except (TypeError, ValueError):
            continue
        conversations[step] = record

    actions_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in actions:
        action = _action_from_record(record)
        actions_by_step[action["step"]].append(action)

    steps = sorted(set(conversations) | set(actions_by_step))
    turns = []
    for step in steps:
        record = conversations.get(step, {})
        turns.append(
            {
                "step": step,
                "content": record.get("content"),
                "thinking": record.get("thinking"),
                "usage": record.get("usage"),
                "actions": actions_by_step.get(step, []),
            }
        )
    return turns


def load_run(run_dir: Path) -> dict[str, Any]:
    run_file = run_dir / "run.json"
    if not run_file.exists():
        raise FileNotFoundError(run_dir)
    metadata = json.loads(run_file.read_text(encoding="utf-8"))

    frames_dir = run_dir / "frames"
    frames = sorted(
        (path.name for path in frames_dir.glob("*.png") if path.is_file())
    ) if frames_dir.exists() else []

    conversation = _read_jsonl(run_dir / "conversation.jsonl")
    actions = _read_jsonl(run_dir / "actions.jsonl")

    return {
        "id": run_dir.name,
        "metadata": metadata,
        "frames": frames,
        "turns": _build_turns(conversation, actions),
    }


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    result = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        run_file = path / "run.json"
        if not run_file.exists():
            continue
        try:
            metadata = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        frames = path / "frames"
        frame_count = (
            len(list(frames.glob("*.png"))) if frames.exists() else 0
        )
        result.append(
            {
                "id": path.name,
                "started_at": metadata.get("started_at"),
                "model": metadata.get("model"),
                "backend": metadata.get("backend"),
                "input": metadata.get("input"),
                "frame_count": frame_count,
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(result, key=lambda item: item["mtime"], reverse=True)


def _safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    if not _RUN_RE.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Unknown run")
    run_dir = (runs_dir / run_id).resolve()
    root = runs_dir.resolve()
    if run_dir.parent != root or not (run_dir / "run.json").exists():
        raise HTTPException(status_code=404, detail="Unknown run")
    return run_dir


def create_app(runs_dir: Path, live: Any | None = None) -> FastAPI:
    runs_dir = runs_dir.expanduser().resolve()
    app = FastAPI(title="ITB Harness Replay")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/api/live")
    def api_live() -> dict[str, Any]:
        if live is None:
            return {"active": False}
        return live.snapshot()

    def _require_live() -> Any:
        if live is None:
            raise HTTPException(status_code=404, detail="No live session")
        return live

    @app.post("/api/live/step")
    def api_live_step() -> dict[str, Any]:
        session = _require_live()
        session.step()
        return session.snapshot()

    @app.post("/api/live/resume")
    def api_live_resume() -> dict[str, Any]:
        session = _require_live()
        session.resume()
        return session.snapshot()

    @app.post("/api/live/pause")
    def api_live_pause() -> dict[str, Any]:
        session = _require_live()
        session.pause()
        return session.snapshot()

    @app.post("/api/live/stop")
    def api_live_stop() -> dict[str, Any]:
        session = _require_live()
        session.stop()
        return session.snapshot()

    @app.get("/api/live/events")
    async def api_live_events(
        after: int = 0,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        session = _require_live()
        if last_event_id:
            try:
                after = max(after, int(last_event_id))
            except ValueError:
                pass

        subscriber = session.subscribe(after)

        async def generate():
            try:
                while True:
                    try:
                        event = await asyncio.to_thread(
                            subscriber.get, True, 1.0
                        )
                    except queue.Empty:
                        snapshot = session.snapshot()
                        yield (
                            f"data: {json.dumps({'type': 'status', **snapshot})}"
                            "\n\n"
                        )
                        continue

                    yield (
                        f"id: {event.get('seq', 0)}\n"
                        f"data: {json.dumps(event)}\n\n"
                    )
                    if event.get("type") in {"stopped", "error"}:
                        break
            finally:
                session.unsubscribe(subscriber)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/runs")
    def api_runs() -> list[dict[str, Any]]:
        return list_runs(runs_dir)

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> dict[str, Any]:
        run_dir = _safe_run_dir(runs_dir, run_id)
        try:
            return load_run(run_dir)
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not read run: {exc}"
            ) from exc

    @app.get("/api/runs/{run_id}/frames/{name}")
    def api_frame(run_id: str, name: str) -> FileResponse:
        run_dir = _safe_run_dir(runs_dir, run_id)
        if not _FRAME_RE.fullmatch(name):
            raise HTTPException(status_code=404, detail="Unknown frame")
        path = run_dir / "frames" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Unknown frame")
        return FileResponse(path, media_type="image/png")

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="itb-web",
        description="Browse recorded ITB harness runs.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("runs"),
        help="Directory containing run directories (default: runs).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    app = create_app(args.runs)
    print(f"Replay viewer: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
