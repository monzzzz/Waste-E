#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from recording_catalog import (
    DEFAULT_RECORDINGS_DIR,
    KNOWN_TOP_LEVEL_ENTRIES,
    resolve_session_dirs,
)

DEFAULT_TARGET_CONFIG = Path(__file__).with_name("runpod_targets.json")
DEFAULT_REMOTE_BASE_PATH = "/workspace/data/waste-e-recordings"
GROUP_ORDER = ("session", "telemetry", "gps", "videos", "composites", "extras")
GROUP_ALIASES = {
    "session": "session",
    "session-json": "session",
    "session.json": "session",
    "telemetry": "telemetry",
    "telemetry.csv": "telemetry",
    "gps": "gps",
    "gps-track": "gps",
    "gps_track": "gps",
    "gps_track.geojson": "gps",
    "videos": "videos",
    "composites": "composites",
    "extra": "extras",
    "extras": "extras",
    "metadata": "metadata",
    "all": "all",
}
GROUP_EXPANSIONS = {
    "session": {"session"},
    "telemetry": {"telemetry"},
    "gps": {"gps"},
    "videos": {"videos"},
    "composites": {"composites"},
    "extras": {"extras"},
    "metadata": {"session", "telemetry", "gps"},
    "all": set(GROUP_ORDER),
}


@dataclass(frozen=True)
class RunPodTarget:
    name: str
    host: str
    user: str
    port: int
    base_path: str
    ssh_key: Path | None = None

    @property
    def remote(self) -> str:
        return f"{self.user}@{self.host}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload selected Waste-E hardware recording sessions to a RunPod over SSH/SCP."
        ),
        epilog=(
            "Examples:\n"
            "  python utils/data/send_recordings_to_runpod.py --target a100 --latest 1\n"
            "  python utils/data/send_recordings_to_runpod.py 2026-05-07_08-11-52 "
            "--target a100 --include metadata videos\n"
            "  python utils/data/send_recordings_to_runpod.py --host 1.2.3.4 "
            "--base-path /workspace/data/run1 --all --dry-run"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "session",
        nargs="*",
        help="Session name(s) under hardware/recordings, or explicit session path(s).",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help=f"Recordings root directory (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload every available session.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        metavar="N",
        help="Upload the latest N sessions. Defaults to 1 when no sessions are given.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        action="append",
        metavar="GROUP",
        help=(
            "Include group(s): session, telemetry, gps, videos, composites, "
            "extras, metadata, all. Repeat or pass multiple names."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        action="append",
        metavar="GROUP",
        help="Exclude group(s) after include expansion.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TARGET_CONFIG,
        help=f"RunPod target config JSON (default: {DEFAULT_TARGET_CONFIG})",
    )
    parser.add_argument(
        "--target",
        help="Named target from the config JSON.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print configured targets and exit.",
    )
    parser.add_argument(
        "--host",
        help="RunPod host or IP address. Overrides config and RUNPOD_HOST.",
    )
    parser.add_argument(
        "--user",
        help="SSH username. Overrides config and RUNPOD_USER.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="SSH port. Overrides config and RUNPOD_PORT.",
    )
    parser.add_argument(
        "--base-path",
        help=(
            "Remote base directory where session folders will land. "
            f"Defaults to {DEFAULT_REMOTE_BASE_PATH}."
        ),
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        help="SSH private key path. Overrides config and RUNPOD_SSH_KEY.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without creating archives or opening SSH.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Leave the uploaded tar.gz archive on the RunPod after extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_targets(args.config)

    if args.list_targets:
        print_targets(targets, config_path=args.config)
        return 0

    target = resolve_target(args, targets)
    groups = resolve_groups(args.include, args.exclude)
    latest = args.latest
    if latest is None and not args.all and not args.session:
        latest = 1

    try:
        sessions = resolve_session_dirs(
            args.session,
            args.recordings_dir,
            select_all=args.all,
            latest=latest,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    require_binary("ssh")
    require_binary("scp")

    uploaded = 0
    for session_dir in sessions:
        selected_paths = collect_selected_paths(session_dir, groups)
        if not selected_paths:
            print(f"[skip] {session_dir.name}: no files matched the requested groups")
            continue

        print(
            f"[session] {session_dir.name} -> {target.remote}:{target.base_path} "
            f"({', '.join(groups)})"
        )
        for selected_path in selected_paths:
            suffix = "/" if selected_path.is_dir() else ""
            print(f"  - {selected_path.relative_to(session_dir)}{suffix}")

        if args.dry_run:
            continue

        upload_session(
            session_dir=session_dir,
            selected_paths=selected_paths,
            target=target,
            keep_archive=args.keep_archive,
        )
        uploaded += 1

    if args.dry_run:
        print(f"[done] dry run covered {len(sessions)} session(s)")
        return 0

    print(f"[done] uploaded {uploaded} session(s) to {target.remote}:{target.base_path}")
    return 0


def load_targets(config_path: Path) -> dict[str, RunPodTarget]:
    if not config_path.exists():
        return {}

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to load target config {config_path}: {exc}")

    if not isinstance(raw, dict):
        raise SystemExit(f"Target config must be a JSON object: {config_path}")

    mapping = raw.get("targets", raw)
    if not isinstance(mapping, dict):
        raise SystemExit(f"Target config must contain a 'targets' object: {config_path}")

    targets: dict[str, RunPodTarget] = {}
    for name, payload in mapping.items():
        if not isinstance(payload, dict):
            raise SystemExit(f"Target '{name}' must be a JSON object.")
        targets[name] = build_target(
            name=name,
            host=_string_value(payload.get("host")),
            user=_string_value(payload.get("user")) or "root",
            port=_int_value(payload.get("port")) or 22,
            base_path=_string_value(payload.get("base_path")) or DEFAULT_REMOTE_BASE_PATH,
            ssh_key=_path_value(payload.get("ssh_key")),
        )
    return targets


def print_targets(targets: dict[str, RunPodTarget], *, config_path: Path) -> None:
    if not targets:
        print(f"No targets configured in {config_path}")
        example_path = config_path.with_name("runpod_targets.example.json")
        if example_path.exists():
            print(f"Example config: {example_path}")
        return

    print(f"Configured targets from {config_path}:")
    for name in sorted(targets):
        target = targets[name]
        key_text = f" key={target.ssh_key}" if target.ssh_key else ""
        print(
            f"  - {name}: {target.user}@{target.host}:{target.port} "
            f"base_path={target.base_path}{key_text}"
        )


def resolve_target(args: argparse.Namespace, targets: dict[str, RunPodTarget]) -> RunPodTarget:
    base_target: RunPodTarget | None = None
    if args.target:
        try:
            base_target = targets[args.target]
        except KeyError as exc:
            raise SystemExit(
                f"Unknown target '{args.target}'. Use --list-targets to see configured names."
            ) from exc
    elif targets and len(targets) == 1 and not args.host:
        base_target = next(iter(targets.values()))

    host = args.host or os.getenv("RUNPOD_HOST") or (base_target.host if base_target else "")
    user = args.user or os.getenv("RUNPOD_USER") or (base_target.user if base_target else "root")
    port = args.port or _int_value(os.getenv("RUNPOD_PORT")) or (base_target.port if base_target else 22)
    base_path = (
        args.base_path
        or os.getenv("RUNPOD_BASE_PATH")
        or (base_target.base_path if base_target else DEFAULT_REMOTE_BASE_PATH)
    )
    ssh_key = (
        args.ssh_key
        or _path_value(os.getenv("RUNPOD_SSH_KEY"))
        or (base_target.ssh_key if base_target else None)
    )
    name = args.target or (base_target.name if base_target else "manual")

    if not host:
        raise SystemExit(
            "No RunPod host provided. Set one with --host, --target, or RUNPOD_HOST."
        )

    return build_target(
        name=name,
        host=host,
        user=user,
        port=port,
        base_path=base_path,
        ssh_key=ssh_key,
    )


def build_target(
    *,
    name: str,
    host: str,
    user: str,
    port: int,
    base_path: str,
    ssh_key: Path | None,
) -> RunPodTarget:
    if not host or not host.strip():
        raise SystemExit(f"Target '{name}' is missing a host.")
    if not base_path or not base_path.strip():
        raise SystemExit(f"Target '{name}' is missing a base_path.")
    if port < 1:
        raise SystemExit("SSH port must be at least 1.")
    return RunPodTarget(
        name=name,
        host=host.strip(),
        user=(user or "root").strip(),
        port=port,
        base_path=base_path.rstrip("/") or "/",
        ssh_key=ssh_key.expanduser().resolve() if ssh_key else None,
    )


def resolve_groups(
    include_args: list[list[str]] | None,
    exclude_args: list[list[str]] | None,
) -> list[str]:
    include_tokens = flatten_group_args(include_args) or ["all"]
    exclude_tokens = flatten_group_args(exclude_args)

    included = expand_group_tokens(include_tokens)
    excluded = expand_group_tokens(exclude_tokens)
    effective = [group for group in GROUP_ORDER if group in included and group not in excluded]
    if not effective:
        raise SystemExit("The include/exclude settings removed every file group.")
    return effective


def flatten_group_args(values: list[list[str]] | None) -> list[str]:
    tokens: list[str] = []
    for group_list in values or []:
        for raw in group_list:
            parts = [part.strip() for part in raw.split(",")]
            tokens.extend(part for part in parts if part)
    return tokens


def expand_group_tokens(tokens: list[str]) -> set[str]:
    expanded: set[str] = set()
    for raw_token in tokens:
        normalized = raw_token.strip().lower().replace("_", "-")
        canonical = GROUP_ALIASES.get(normalized)
        if canonical is None:
            valid = ", ".join(sorted(GROUP_ALIASES))
            raise SystemExit(f"Unknown group '{raw_token}'. Valid group names: {valid}")
        expanded.update(GROUP_EXPANSIONS[canonical])
    return expanded


def collect_selected_paths(session_dir: Path, groups: list[str]) -> list[Path]:
    selected: list[Path] = []
    for group in groups:
        if group == "session":
            selected.extend(_existing_paths(session_dir / "session.json"))
        elif group == "telemetry":
            selected.extend(_existing_paths(session_dir / "telemetry.csv"))
        elif group == "gps":
            selected.extend(_existing_paths(session_dir / "gps_track.geojson"))
        elif group == "videos":
            selected.extend(_existing_paths(session_dir / "videos"))
        elif group == "composites":
            selected.extend(_existing_paths(session_dir / "composites"))
        elif group == "extras":
            for path in sorted(session_dir.iterdir()):
                if path.name not in KNOWN_TOP_LEVEL_ENTRIES:
                    selected.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def upload_session(
    *,
    session_dir: Path,
    selected_paths: list[Path],
    target: RunPodTarget,
    keep_archive: bool,
) -> None:
    incoming_dir = posixpath.join(target.base_path, ".incoming")
    archive_name = f"{session_dir.name}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="wastee-runpod-upload-") as temp_dir:
        archive_path = Path(temp_dir) / archive_name
        create_archive(session_dir, selected_paths, archive_path)

        remote_archive = posixpath.join(incoming_dir, archive_name)
        run_remote_command(
            target,
            f"mkdir -p {shlex.quote(target.base_path)} {shlex.quote(incoming_dir)}",
        )
        run_scp_upload(target, archive_path, remote_archive)

        extract_command = (
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(target.base_path)}"
        )
        if not keep_archive:
            extract_command += f" && rm -f {shlex.quote(remote_archive)}"
        run_remote_command(target, extract_command)


def create_archive(session_dir: Path, selected_paths: list[Path], archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in selected_paths:
            relative_path = path.relative_to(session_dir)
            archive.add(path, arcname=str(Path(session_dir.name) / relative_path))


def run_remote_command(target: RunPodTarget, remote_command: str) -> None:
    command = ["ssh", "-p", str(target.port)]
    if target.ssh_key:
        command.extend(["-i", str(target.ssh_key)])
    command.append(target.remote)
    command.append(remote_command)
    run_command(command)


def run_scp_upload(target: RunPodTarget, local_path: Path, remote_path: str) -> None:
    command = ["scp", "-P", str(target.port)]
    if target.ssh_key:
        command.extend(["-i", str(target.ssh_key)])
    command.extend([str(local_path), f"{target.remote}:{remote_path}"])
    run_command(command)


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return

    output = (result.stderr or result.stdout).strip()
    if not output:
        output = f"{command[0]} exited with status {result.returncode}"
    raise SystemExit(f"{' '.join(command)}\n{output}")


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary not found on PATH: {name}")


def _existing_paths(path: Path) -> list[Path]:
    return [path] if path.exists() else []


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_value(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"Expected an integer value, got {value!r}")


def _path_value(value: object) -> Path | None:
    text = _string_value(value)
    return Path(text).expanduser() if text else None


if __name__ == "__main__":
    sys.exit(main())
