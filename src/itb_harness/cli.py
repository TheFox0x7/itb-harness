"""Small REPL for the observe side of the harness.

This is intentionally not the agent loop yet.  It gives us a trustworthy
"look at the game window and write a frame to disk" primitive that the model
loop can call later.
"""

from __future__ import annotations

import argparse
import cmd
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .capture import (
    BackendUnavailable,
    CaptureBackend,
    CaptureError,
    Frame,
    WindowInfo,
    WindowNotFound,
    get_backend,
    resolve_window,
)


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)].rstrip() + "..."


def format_windows(
    windows: list[WindowInfo], target_id: str | None = None
) -> str:
    if not windows:
        return "No windows found."

    rows: list[str] = []
    for index, window in enumerate(windows, start=1):
        selected = "*" if window.id == target_id else " "
        flags = []
        if window.active:
            flags.append("A")
        if window.fullscreen:
            flags.append("F")
        if window.minimized:
            flags.append("m")
        flag_text = "".join(flags) or "-"
        rows.append(
            f"{selected}{index:2d}  "
            f"{(window.monitor or '?'):6}  "
            f"{window.width:4d}x{window.height:<4d}  "
            f"{flag_text:3}  "
            f"{_truncate(window.app_id, 22):22}  "
            f"{_truncate(window.label, 68)}"
        )

    header = (
        "    #  output   geometry      flg  app_id                  title"
    )
    return "\n".join([header, *rows])


class ObservationRun:
    """One session's target window and frame output."""

    def __init__(self, backend: CaptureBackend, out_dir: Path) -> None:
        self.backend = backend
        self.target: WindowInfo | None = None
        self.frame_index = 0

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = out_dir / stamp
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.root / "manifest.jsonl"
        self._manifest = self.manifest_path.open("a", encoding="utf-8")

        monitors = [asdict(monitor) for monitor in backend.list_monitors()]
        (self.root / "run.json").write_text(
            json.dumps(
                {
                    "backend": backend.name,
                    "started_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "monitors": monitors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def resolve(self, spec: str) -> WindowInfo:
        return resolve_window(self.backend.list_windows(), spec)

    def set_target(self, spec: str) -> WindowInfo:
        window = self.resolve(spec)
        self.target = window
        print(f"Target: [{window.monitor or '?'}] {window.label}")
        print(f"        {window.width}x{window.height} at {window.x},{window.y}")
        return window

    def _refresh_target(self) -> WindowInfo:
        if self.target is None:
            raise CaptureError("No target selected. Use `target <index|name>`.")
        window_id = self.target.id
        for window in self.backend.list_windows():
            if window.id == window_id:
                self.target = window
                return window
        raise WindowNotFound(
            f"Target window closed: {self.target.label!r}. Pick another with `target`."
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def capture(self) -> Frame:
        window = self._refresh_target()
        frame = self.backend.capture_window(window.id)

        self.frame_index += 1
        path = self.frames_dir / f"{self.frame_index:04d}.png"
        path.write_bytes(frame.png)

        record = {
            "frame": self.frame_index,
            "file": str(path.relative_to(self.root)),
            "captured_at": frame.captured_at,
            "latency_ms": round(frame.latency_ms, 2),
            "width": frame.width,
            "height": frame.height,
            "window": asdict(frame.window),
        }
        self._manifest.write(json.dumps(record) + "\n")
        self._manifest.flush()

        print(
            f"[{self.frame_index:04d}] "
            f"{frame.width}x{frame.height}  "
            f"{frame.window.monitor or '?':6}  "
            f"{frame.latency_ms:6.1f} ms  "
            f"{path}"
        )
        return frame

    def close(self) -> None:
        self._manifest.close()

    def __enter__(self) -> "ObservationRun":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class Repl(cmd.Cmd):
    intro = (
        "Observation REPL. `windows` to list, `target N` to pick, "
        "`observe` to save a frame, `help` for commands."
    )
    prompt = "itb> "

    def __init__(self, run: ObservationRun, interval: float = 1.0) -> None:
        super().__init__()
        self.run = run
        self.interval = interval

    def _print_windows(self) -> None:
        windows = self.run.backend.list_windows()
        target_id = self.run.target.id if self.run.target else None
        print(format_windows(windows, target_id))

    def do_windows(self, arg: str) -> None:
        """List windows visible to the capture backend."""
        try:
            self._print_windows()
        except CaptureError as exc:
            print(f"error: {exc}")

    do_w = do_windows
    do_ls = do_windows

    def do_monitors(self, arg: str) -> None:
        """List monitors/outputs and their geometry."""
        try:
            for monitor in self.run.backend.list_monitors():
                print(
                    f"{monitor.name:8} {monitor.width}x{monitor.height} "
                    f"at {monitor.x},{monitor.y} (scale {monitor.scale:g})"
                )
        except CaptureError as exc:
            print(f"error: {exc}")

    def do_target(self, arg: str) -> None:
        """Select the window to capture: `target 3` or `target Breach`."""
        if not arg.strip():
            if self.run.target:
                window = self.run.target
                print(f"[{window.monitor or '?'}] {window.label}")
            else:
                print("No target selected.")
            return
        try:
            self.run.set_target(arg)
        except (ValueError, CaptureError) as exc:
            print(f"error: {exc}")

    do_t = do_target

    def do_observe(self, arg: str) -> None:
        """Capture the target window and save a PNG frame."""
        try:
            self.run.capture()
        except CaptureError as exc:
            print(f"error: {exc}")

    do_s = do_observe

    def do_auto(self, arg: str) -> None:
        """Capture repeatedly: `auto [count] [interval-seconds]`."""
        parts = arg.split()
        count = 1
        interval = self.interval
        try:
            if parts:
                count = int(parts[0])
            if len(parts) > 1:
                interval = float(parts[1])
        except ValueError:
            print("usage: auto [count] [interval-seconds]")
            return

        try:
            for _ in range(max(0, count)):
                self.run.capture()
                if interval > 0:
                    time.sleep(interval)
        except KeyboardInterrupt:
            print("\nstopped")
        except CaptureError as exc:
            print(f"error: {exc}")

    def do_status(self, arg: str) -> None:
        """Show backend, target, output directory, and frame count."""
        print(f"backend: {self.run.backend.name}")
        print(f"output:  {self.run.root}")
        print(f"frames:  {self.run.frame_index}")
        if self.run.target:
            window = self.run.target
            print(
                f"target:  [{window.monitor or '?'}] {window.label} "
                f"({window.width}x{window.height} at {window.x},{window.y})"
            )
        else:
            print("target:  none")

    def do_quit(self, arg: str) -> bool:
        """Exit the REPL."""
        return True

    do_q = do_quit
    do_exit = do_quit

    def default(self, line: str) -> None:
        print(f"unknown command: {line!r} (try `help`)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itb-harness",
        description="Capture a game window and record observation frames.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Capture backend: auto (default), kwin, or x11.",
    )
    parser.add_argument(
        "-w",
        "--window",
        help="Window to target: index, UUID, app id, or title substring.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="Directory for run output (default: runs).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one frame and exit (requires --window).",
    )
    parser.add_argument(
        "--list",
        dest="list_windows",
        action="store_true",
        help="List windows and exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Default `auto` interval in seconds (default: 1.0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        backend = get_backend(args.backend)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.list_windows:
            print(format_windows(backend.list_windows()))
            return 0

        with ObservationRun(backend, args.out.expanduser()) as run:
            if args.window:
                try:
                    run.set_target(args.window)
                except (ValueError, CaptureError) as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2

            if args.once:
                if run.target is None:
                    print("error: --once requires --window", file=sys.stderr)
                    return 2
                try:
                    run.capture()
                except CaptureError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                return 0

            repl = Repl(run, interval=args.interval)
            if run.target is None:
                try:
                    print(format_windows(backend.list_windows()))
                except CaptureError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
            repl.cmdloop()
            return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
