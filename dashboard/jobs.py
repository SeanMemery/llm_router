from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from paths import ArtifactLayout, get_artifact_layout
from patterns.paths import PATTERN_LIBRARY_FILENAME


JobStatus = Literal["queued", "running", "paused", "succeeded", "failed"]
QUEUEABLE_JOB_ACTIONS = {
    "supervised-learn-and-evaluate",
    "unsupervised-launch",
    "evaluate-deepphy",
}
AUTO_RESUME_JOB_ACTIONS = set(QUEUEABLE_JOB_ACTIONS)
AUTO_RESUME_DEFAULT_MAX_ATTEMPTS = 20
AUTO_RESUME_DEFAULT_BACKOFF_SECONDS = 15.0
AUTO_RESUME_DEFAULT_BACKOFF_CAP_SECONDS = 300.0
AUTO_RESUME_PAUSE_REASON = "auto_retry_wait"
ROUTER_WAIT_PAUSE_REASON = "router_wait"
RESOURCE_FAILURE_EXIT_CODES = {-15, -9, 137, 143, 247}
RESOURCE_FAILURE_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "memoryerror",
    "cannot allocate memory",
)
TRANSIENT_LLM_FAILURE_PATTERNS = (
    "no active backend connections",
    "no active llm connection",
    "connection refused",
    "connecterror",
    "connecttimeout",
    "readtimeout",
    "read timeout",
    "remoteprotocolerror",
    "remotedisconnected",
    "remote end closed connection without response",
    "server disconnected without sending a response",
    "temporarily unavailable",
    "temporary failure in name resolution",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "429 too many requests",
    "http 500 from http://127.0.0.1:8787/v1/chat/completions",
    "http 500 from http://0.0.0.0:8787/v1/chat/completions",
)
CURRENT_FULL_TEMPLATE_ITERATION_COUNT = 1
CURRENT_PATTERN_PATIENCE_FACTOR = 0.10
_QUEUE_DISPATCH_PRIORITY = {
    "supervised-learn-and-evaluate": 0,
    "evaluate-deepphy": 0,
    "unsupervised-launch": 1,
}


@dataclass(frozen=True)
class DashboardJobSnapshot:
    job_id: str
    name: str
    status: JobStatus
    argv: list[str]
    metadata: dict[str, Any]
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    log_path: Path
    stdout_path: Path
    stderr_path: Path
    snapshot_path: Path
    recent_output: list[str]
    pid: int | None = None
    pid_token: str | None = None
    cwd: Path | None = None
    env: dict[str, str] | None = None


@dataclass
class _JobRecord:
    snapshot: DashboardJobSnapshot
    process: subprocess.Popen[str] | None
    tail: deque[str]
    cwd: Path | None
    env: dict[str, str] | None
    pause_requested: bool = False
    requeue_requested: bool = False


class DashboardJobManager:
    def __init__(
        self,
        *,
        layout: ArtifactLayout | None = None,
        jobs_root: Path | None = None,
        python_executable: str | None = None,
        max_recent_lines: int = 200,
        max_concurrent_queue_jobs: int = 1,
    ) -> None:
        resolved_layout = layout or get_artifact_layout()
        self._jobs_root = jobs_root or (resolved_layout.data_root / "dashboard_jobs")
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._python_executable = python_executable or sys.executable
        self._max_recent_lines = max_recent_lines
        self._max_concurrent_queue_jobs = max(1, int(max_concurrent_queue_jobs))
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobRecord] = {}
        self._dispatching_job_ids: set[str] = set()
        self._load_existing_jobs()
        self._maintenance_stop = threading.Event()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
        )
        self._maintenance_thread.start()
        self._schedule_next_queued_job()

    @property
    def max_concurrent_queue_jobs(self) -> int:
        with self._lock:
            return self._max_concurrent_queue_jobs

    def set_max_concurrent_queue_jobs(self, value: int) -> int:
        normalized = max(1, int(value))
        with self._lock:
            self._max_concurrent_queue_jobs = normalized
        self._schedule_next_queued_job()
        return normalized

    def launch(
        self,
        *,
        name: str,
        argv: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        enqueue: bool = False,
    ) -> DashboardJobSnapshot:
        if not argv:
            raise ValueError("argv must not be empty")

        job_id = uuid.uuid4().hex
        now = _utc_now()
        job_root = self._jobs_root / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        log_path = job_root / "job.log"
        stdout_path = job_root / "stdout.log"
        stderr_path = job_root / "stderr.log"
        snapshot_path = job_root / "snapshot.json"
        command = self._normalize_argv(argv)
        initial_metadata = self._initial_metadata(
            {
                **dict(metadata or {}),
            },
            argv=command,
        ) | {
            "queue_requested": enqueue,
            "launch_mode": "queue" if enqueue else "run_now",
        }
        with self._lock:
            if enqueue and initial_metadata.get("action") in QUEUEABLE_JOB_ACTIONS:
                initial_metadata.setdefault(
                    "queue_order",
                    self._next_queue_order_locked(),
                )
            conflict_message = self._unsupervised_queue_conflict_message_locked(
                initial_metadata
            )
            if conflict_message is not None:
                raise ValueError(conflict_message)
            target_conflict_message = self._queueable_target_conflict_message_locked(
                initial_metadata
            )
            if target_conflict_message is not None:
                raise ValueError(target_conflict_message)
        initial = DashboardJobSnapshot(
            job_id=job_id,
            name=name,
            status="queued",
            argv=command,
            metadata=initial_metadata,
            created_at=now,
            started_at=None,
            finished_at=None,
            exit_code=None,
            log_path=log_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            snapshot_path=snapshot_path,
            recent_output=[],
            pid=None,
            cwd=cwd,
            env=env,
        )
        record = _JobRecord(
            snapshot=initial,
            process=None,
            tail=deque(maxlen=self._max_recent_lines),
            cwd=cwd,
            env=env,
        )
        self._write_snapshot(initial)
        with self._lock:
            self._jobs[job_id] = record
        if enqueue:
            self._schedule_next_queued_job()
            return self.get_snapshot(job_id)
        return self._start_job(job_id)

    def _unsupervised_queue_conflict_message_locked(
        self,
        metadata: dict[str, Any],
        *,
        exclude_job_id: str | None = None,
    ) -> str | None:
        if metadata.get("action") != "unsupervised-launch":
            return None
        if not bool(metadata.get("queue_requested")):
            return None
        env_id = metadata.get("env_id")
        if not isinstance(env_id, str) or not env_id.strip():
            return None
        for record in self._jobs.values():
            snapshot = record.snapshot
            if exclude_job_id is not None and snapshot.job_id == exclude_job_id:
                continue
            existing_metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
            if existing_metadata.get("action") != "unsupervised-launch":
                continue
            if not bool(existing_metadata.get("queue_requested")):
                continue
            if existing_metadata.get("env_id") != env_id:
                continue
            if snapshot.status not in {"queued", "running", "paused"}:
                continue
            return (
                "Only one queue-backed unsupervised job is allowed per environment. "
                f"{env_id} already has {snapshot.job_id} in state {snapshot.status}."
            )
        return None

    def _queueable_target_conflict_message_locked(
        self,
        metadata: dict[str, Any],
        *,
        exclude_job_id: str | None = None,
    ) -> str | None:
        identity = self._queueable_target_identity(metadata)
        if identity is None:
            return None
        for record in self._jobs.values():
            snapshot = record.snapshot
            if exclude_job_id is not None and snapshot.job_id == exclude_job_id:
                continue
            if snapshot.status not in {"queued", "running", "paused"}:
                continue
            existing_metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
            if self._queueable_target_identity(existing_metadata) != identity:
                continue
            return (
                "Only one queued or active dashboard job is allowed per run target. "
                f"{snapshot.job_id} already targets {identity[1]} in state {snapshot.status}."
            )
        return None

    def _queueable_target_identity(
        self, metadata: dict[str, Any]
    ) -> tuple[str, str] | None:
        action = metadata.get("action")
        if action not in QUEUEABLE_JOB_ACTIONS:
            return None
        for key in ("evaluation_run_dir", "run_dir"):
            raw_value = metadata.get(key)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            normalized = raw_value.strip()
            try:
                normalized = str(Path(normalized).resolve())
            except OSError:
                pass
            return (key, normalized)
        return None

    def _snapshot_target_identity(
        self, snapshot: DashboardJobSnapshot
    ) -> tuple[str, str] | None:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        return self._queueable_target_identity(metadata)

    def _is_superseded_by_newer_succeeded_target_locked(
        self, snapshot: DashboardJobSnapshot
    ) -> bool:
        identity = self._snapshot_target_identity(snapshot)
        if identity is None:
            return False
        current_created_at = snapshot.created_at or ""
        for record in self._jobs.values():
            other = record.snapshot
            if other.job_id == snapshot.job_id:
                continue
            if other.status != "succeeded":
                continue
            if self._snapshot_target_identity(other) != identity:
                continue
            if (other.created_at or "") > current_created_at:
                return True
        return False

    def get_snapshot(self, job_id: str) -> DashboardJobSnapshot:
        self._refresh_completion(job_id)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            return self._replace_snapshot(
                record.snapshot, recent_output=list(record.tail)
            )

    def list_snapshots(self) -> list[DashboardJobSnapshot]:
        with self._lock:
            job_ids = list(self._jobs)
        snapshots = [self.get_snapshot(job_id) for job_id in job_ids]
        return sorted(snapshots, key=lambda item: item.created_at, reverse=True)

    def read_recent_output(self, job_id: str, *, limit: int = 80) -> list[str]:
        snapshot = self.get_snapshot(job_id)
        if limit <= 0:
            return []
        return snapshot.recent_output[-limit:]

    def wait_for_completion(
        self, job_id: str, *, timeout: float | None = None
    ) -> DashboardJobSnapshot:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            process = record.process
        if process is not None:
            process.wait(timeout=timeout)
        return self.get_snapshot(job_id)

    def pause(
        self,
        job_id: str,
        *,
        reason: str = "manual",
    ) -> DashboardJobSnapshot:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            if not record.snapshot.metadata.get("supports_pause", False):
                raise ValueError(f"Dashboard job does not support pause: {job_id}")
            if record.snapshot.status not in {"queued", "running"}:
                raise ValueError(
                    f"Dashboard job is not pauseable in state {record.snapshot.status}: {job_id}"
                )
            process = record.process
            pid = record.snapshot.pid
            record.pause_requested = True
            metadata = self._paused_metadata(record.snapshot.metadata, reason=reason)
            recent_output = self._append_dashboard_note_locked(
                record,
                self._pause_note(reason),
            )
            updated = self._replace_snapshot(
                record.snapshot,
                status="paused",
                metadata=metadata,
                finished_at=None,
                exit_code=None,
                recent_output=recent_output,
            )
            record.snapshot = updated
        self._write_snapshot(updated)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                return self.get_snapshot(job_id)
            self._refresh_completion(job_id)
            return self.get_snapshot(job_id)
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return updated

    def resume(self, job_id: str) -> DashboardJobSnapshot:
        self._refresh_completion(job_id)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            supports_resume = record.snapshot.metadata.get("supports_pause", False) or (
                isinstance(record.snapshot.metadata.get("resume_argv"), list)
                and bool(record.snapshot.metadata.get("resume_argv"))
            )
            if not supports_resume:
                raise ValueError(f"Dashboard job does not support resume: {job_id}")
            if record.snapshot.status not in {"paused", "failed"}:
                raise ValueError(
                    f"Dashboard job is not resumable in state {record.snapshot.status}: {job_id}"
                )
            if record.process is not None and record.process.poll() is None:
                raise ValueError(f"Dashboard job has not fully paused yet: {job_id}")
            resume_argv = record.snapshot.metadata.get("resume_argv")
        if isinstance(resume_argv, list) and resume_argv:
            return self._start_job(job_id, argv_override=[str(part) for part in resume_argv])
        return self._start_job(job_id)

    def requeue(self, job_id: str) -> DashboardJobSnapshot:
        self._refresh_completion(job_id)
        process: subprocess.Popen[str] | None = None
        pid: int | None = None
        queued_snapshot: DashboardJobSnapshot | None = None
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            snapshot = record.snapshot
            metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
            supports_resume = metadata.get("supports_pause", False) or (
                isinstance(metadata.get("resume_argv"), list)
                and bool(metadata.get("resume_argv"))
            )
            if metadata.get("action") not in QUEUEABLE_JOB_ACTIONS:
                raise ValueError(f"Dashboard job does not use the launch queue: {job_id}")
            if not bool(metadata.get("queue_requested")):
                raise ValueError(f"Dashboard job was not launched in queue mode: {job_id}")
            if not supports_resume:
                raise ValueError(f"Dashboard job does not support queue re-entry: {job_id}")
            if snapshot.status not in {"running", "paused"}:
                raise ValueError(
                    f"Dashboard job is not requeueable in state {snapshot.status}: {job_id}"
                )
            if snapshot.status == "paused":
                updated = self._transition_record_to_queued_locked(
                    record,
                    note="Moved back to the queue.",
                )
                queued_snapshot = updated
            else:
                record.requeue_requested = True
                recent_output = self._append_dashboard_note_locked(
                    record,
                    "Stopping job and moving it to the queue tail.",
                )
                record.snapshot = self._replace_snapshot(
                    snapshot,
                    recent_output=recent_output,
                )
                process = record.process
                pid = snapshot.pid
        if queued_snapshot is not None:
            self._write_snapshot(queued_snapshot)
            self._schedule_next_queued_job()
            return queued_snapshot
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5.0)
            return self.get_snapshot(job_id)
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.time() + 5.0
            while time.time() < deadline and self._pid_is_alive(pid):
                time.sleep(0.05)
            return self.get_snapshot(job_id)
        return self.get_snapshot(job_id)

    def reorder_queue(self, ordered_job_ids: list[str]) -> list[DashboardJobSnapshot]:
        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for raw_job_id in ordered_job_ids:
            job_id = str(raw_job_id).strip()
            if not job_id or job_id in seen_ids:
                continue
            normalized_ids.append(job_id)
            seen_ids.add(job_id)
        with self._lock:
            pending_job_ids = [
                snapshot.job_id
                for snapshot in sorted(
                    (record.snapshot for record in self._jobs.values() if self._is_pending_queue_job(record.snapshot)),
                    key=self._queue_sort_key,
                )
            ]
            if set(normalized_ids) != set(pending_job_ids):
                raise ValueError("Queue reorder payload must include every queued job exactly once.")
            updated_snapshots: list[DashboardJobSnapshot] = []
            for index, job_id in enumerate(normalized_ids, start=1):
                record = self._jobs[job_id]
                metadata = dict(record.snapshot.metadata)
                metadata["queue_order"] = index
                updated = self._replace_snapshot(record.snapshot, metadata=metadata)
                record.snapshot = updated
                updated_snapshots.append(updated)
        for snapshot in updated_snapshots:
            self._write_snapshot(snapshot)
        return [self.get_snapshot(job_id) for job_id in normalized_ids]

    def clear(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            if record.snapshot.status not in {"failed", "succeeded"}:
                raise ValueError(
                    f"Dashboard job is not clearable in state {record.snapshot.status}: {job_id}"
                )
            job_root = record.snapshot.snapshot_path.parent
            del self._jobs[job_id]
        shutil.rmtree(job_root, ignore_errors=True)
        self._schedule_next_queued_job()

    def clear_all_failed(self) -> int:
        with self._lock:
            failed_job_ids = [
                job_id
                for job_id, record in self._jobs.items()
                if record.snapshot.status == "failed"
            ]
            job_roots = [
                self._jobs[job_id].snapshot.snapshot_path.parent for job_id in failed_job_ids
            ]
            for job_id in failed_job_ids:
                del self._jobs[job_id]
        for job_root in job_roots:
            shutil.rmtree(job_root, ignore_errors=True)
        self._schedule_next_queued_job()
        return len(failed_job_ids)

    def terminate_and_remove(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            process = record.process
            pid = record.snapshot.pid
            job_root = record.snapshot.snapshot_path.parent
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5.0)
        elif pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                del self._jobs[job_id]
        shutil.rmtree(job_root, ignore_errors=True)
        self._schedule_next_queued_job()

    def exit_paused(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            if record.snapshot.status != "paused":
                raise ValueError(
                    f"Dashboard job is not exitable in state {record.snapshot.status}: {job_id}"
                )
            process = record.process
            pid = record.snapshot.pid
            if process is not None and process.poll() is None:
                raise ValueError(f"Dashboard job is still stopping: {job_id}")
            if pid is not None and self._pid_is_alive(pid):
                raise ValueError(f"Dashboard job is still running: {job_id}")
            job_root = record.snapshot.snapshot_path.parent
            del self._jobs[job_id]
        shutil.rmtree(job_root, ignore_errors=True)
        self._schedule_next_queued_job()

    def _normalize_argv(self, argv: list[str]) -> list[str]:
        normalized = list(argv)
        first_token = normalized[0]
        first = Path(first_token).name
        if first_token == first and first.startswith("python"):
            normalized[0] = self._python_executable
            return normalized
        if normalized[0].startswith("-"):
            return [self._python_executable, *normalized]
        return normalized

    def _refresh_completion(self, job_id: str) -> None:
        self._refresh_recent_output(job_id)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            process = record.process
            snapshot = record.snapshot
        if process is not None:
            code = process.poll()
            if code is None:
                return
            self._mark_terminal(job_id, code)
            return
        if snapshot.status == "queued":
            return
        if snapshot.status != "running":
            return
        if self._pid_matches_snapshot(snapshot):
            return
        self._mark_terminal(job_id, self._read_exit_code(snapshot))

    def _start_job(
        self, job_id: str, *, argv_override: list[str] | None = None
    ) -> DashboardJobSnapshot:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(f"Unknown dashboard job: {job_id}")
            snapshot = record.snapshot
            cwd = record.cwd
            env = dict(os.environ)
            if record.env is not None:
                env.update(record.env)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
        existing_supervisors = self._find_live_supervisor_pids(snapshot)
        if existing_supervisors:
            note = (
                "Found existing live supervisor"
                + ("s" if len(existing_supervisors) != 1 else "")
                + " for this job while starting; terminating stale tree"
                + ("s" if len(existing_supervisors) != 1 else "")
                + "."
            )
            self._append_dashboard_note(snapshot.log_path, note)
            for pid in existing_supervisors:
                self._terminate_supervisor_pid(pid)
        self._remove_failure_snapshot(snapshot.snapshot_path.parent)
        normalized_argv = self._normalize_argv(argv_override or snapshot.argv)
        popen = subprocess.Popen(
            self._supervisor_argv(snapshot, normalized_argv),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            start_new_session=True,
        )
        running_metadata = self._running_metadata(snapshot.metadata)
        running = self._replace_snapshot(
            snapshot,
            status="running",
            argv=normalized_argv,
            metadata=running_metadata,
            started_at=_utc_now(),
            finished_at=None,
            exit_code=None,
            recent_output=self._read_recent_output(snapshot.log_path),
            pid=popen.pid,
            pid_token=self._pid_token(popen.pid),
        )
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                popen.terminate()
                raise KeyError(f"Unknown dashboard job: {job_id}")
            record.process = popen
            record.pause_requested = False
            record.snapshot = running
            self._dispatching_job_ids.discard(job_id)
        self._write_snapshot(running)
        threading.Thread(
            target=self._monitor_completion, args=(job_id,), daemon=True
        ).start()
        return running

    def _remove_failure_snapshot(self, job_root: Path) -> None:
        failure_snapshot_path = job_root / "failed_resource_snapshot.json"
        try:
            failure_snapshot_path.unlink()
        except FileNotFoundError:
            pass

    def _monitor_completion(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            process = record.process if record is not None else None
        if process is None:
            return
        code = process.wait()
        self._mark_terminal(job_id, code)

    def _refresh_recent_output(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            lines = self._read_recent_output(record.snapshot.log_path)
            record.tail = deque(lines, maxlen=self._max_recent_lines)
            updated = self._replace_snapshot(
                record.snapshot, recent_output=list(record.tail)
            )
            record.snapshot = updated
        self._write_snapshot(updated)

    def _mark_terminal(self, job_id: str, exit_code: int | None) -> None:
        schedule_queue = False
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if record.snapshot.status in {"succeeded", "failed"}:
                return
            if record.pause_requested:
                updated = self._replace_snapshot(
                    record.snapshot,
                    status="paused",
                    exit_code=None,
                    finished_at=_utc_now(),
                    recent_output=list(record.tail),
                    pid=None,
                    pid_token=None,
                )
                record.snapshot = updated
                record.process = None
                record.pause_requested = False
                self._write_snapshot(updated)
                return
            if record.requeue_requested:
                updated = self._transition_record_to_queued_locked(
                    record,
                    note="Moved back to the queue.",
                )
                schedule_queue = True
            else:
                if exit_code != 0:
                    retry_reason = self._classify_retryable_failure(
                        record.snapshot,
                        exit_code=exit_code,
                    )
                    if retry_reason is not None:
                        updated = self._transition_to_auto_resume_pause_locked(
                            record,
                            exit_code=exit_code,
                            retry_reason=retry_reason,
                        )
                        self._write_snapshot(updated)
                        return
                status: JobStatus = "succeeded" if exit_code == 0 else "failed"
                metadata = self._terminal_metadata(
                    record.snapshot.metadata,
                    succeeded=exit_code == 0,
                )
                updated = self._replace_snapshot(
                    record.snapshot,
                    status=status,
                    metadata=metadata,
                    exit_code=exit_code,
                    finished_at=_utc_now(),
                    recent_output=list(record.tail),
                    pid=None,
                    pid_token=None,
                )
                record.snapshot = updated
                record.process = None
        self._write_snapshot(updated)
        if schedule_queue:
            self._schedule_next_queued_job()
            return
        self._schedule_next_queued_job()

    def _replace_snapshot(
        self, snapshot: DashboardJobSnapshot, **changes: Any
    ) -> DashboardJobSnapshot:
        data = {
            "job_id": snapshot.job_id,
            "name": snapshot.name,
            "status": snapshot.status,
            "argv": snapshot.argv,
            "metadata": snapshot.metadata,
            "created_at": snapshot.created_at,
            "started_at": snapshot.started_at,
            "finished_at": snapshot.finished_at,
            "exit_code": snapshot.exit_code,
            "log_path": snapshot.log_path,
            "stdout_path": snapshot.stdout_path,
            "stderr_path": snapshot.stderr_path,
            "snapshot_path": snapshot.snapshot_path,
            "recent_output": snapshot.recent_output,
            "pid": snapshot.pid,
            "pid_token": snapshot.pid_token,
            "cwd": snapshot.cwd,
            "env": snapshot.env,
        }
        data.update(changes)
        return DashboardJobSnapshot(**data)

    def _write_snapshot(self, snapshot: DashboardJobSnapshot) -> None:
        payload = {
            "job_id": snapshot.job_id,
            "name": snapshot.name,
            "status": snapshot.status,
            "argv": snapshot.argv,
            "metadata": snapshot.metadata,
            "created_at": snapshot.created_at,
            "started_at": snapshot.started_at,
            "finished_at": snapshot.finished_at,
            "exit_code": snapshot.exit_code,
            "log_path": str(snapshot.log_path),
            "stdout_path": str(snapshot.stdout_path),
            "stderr_path": str(snapshot.stderr_path),
            "snapshot_path": str(snapshot.snapshot_path),
            "recent_output": snapshot.recent_output,
            "pid": snapshot.pid,
            "pid_token": snapshot.pid_token,
            "cwd": str(snapshot.cwd) if snapshot.cwd is not None else None,
            "env": snapshot.env,
        }
        snapshot.snapshot_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _load_existing_jobs(self) -> None:
        loaded: dict[str, _JobRecord] = {}
        for snapshot_path in sorted(self._jobs_root.glob("*/snapshot.json")):
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                snapshot = DashboardJobSnapshot(
                    job_id=str(payload["job_id"]),
                    name=str(payload["name"]),
                    status=str(payload["status"]),
                    argv=list(payload["argv"]),
                    metadata=dict(payload.get("metadata") or {}),
                    created_at=str(payload["created_at"]),
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    exit_code=payload.get("exit_code"),
                    log_path=Path(payload["log_path"]),
                    stdout_path=Path(payload["stdout_path"]),
                    stderr_path=Path(payload["stderr_path"]),
                    snapshot_path=Path(payload["snapshot_path"]),
                    recent_output=list(payload.get("recent_output") or []),
                    pid=payload.get("pid"),
                    pid_token=payload.get("pid_token"),
                    cwd=Path(payload["cwd"]) if payload.get("cwd") else None,
                    env=dict(payload["env"]) if isinstance(payload.get("env"), dict) else None,
                )
            except (KeyError, TypeError, ValueError):
                continue
            snapshot = self._normalize_queued_learning_snapshot(snapshot)
            snapshot = self._recover_snapshot(snapshot)
            tail = deque(self._read_recent_output(snapshot.log_path), maxlen=self._max_recent_lines)
            snapshot = self._replace_snapshot(snapshot, recent_output=list(tail))
            loaded[snapshot.job_id] = _JobRecord(
                snapshot=snapshot,
                process=None,
                tail=tail,
                cwd=snapshot.cwd,
                env=snapshot.env,
            )
        self._ensure_queue_orders(loaded)
        with self._lock:
            self._jobs = loaded

    def _normalize_queued_learning_snapshot(
        self, snapshot: DashboardJobSnapshot
    ) -> DashboardJobSnapshot:
        if snapshot.status != "queued":
            return snapshot
        metadata = dict(snapshot.metadata) if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("action") not in {"supervised-learn-and-evaluate", "unsupervised-launch"}:
            return snapshot
        if metadata.get("funsearch_template") != "full":
            return snapshot

        request = dict(metadata.get("request") or {})
        updated_argv = _replace_flag_value(
            snapshot.argv,
            "--iteration-count",
            str(CURRENT_FULL_TEMPLATE_ITERATION_COUNT),
        )
        updated_resume_argv = _replace_flag_value(
            metadata.get("resume_argv"),
            "--iteration-count",
            str(CURRENT_FULL_TEMPLATE_ITERATION_COUNT),
        )

        changed = False
        if request.get("iteration_count") != CURRENT_FULL_TEMPLATE_ITERATION_COUNT:
            request["iteration_count"] = CURRENT_FULL_TEMPLATE_ITERATION_COUNT
            changed = True
        if request.get("patience_factor") != CURRENT_PATTERN_PATIENCE_FACTOR:
            request["patience_factor"] = CURRENT_PATTERN_PATIENCE_FACTOR
            changed = True
        if updated_argv != snapshot.argv:
            changed = True
        if updated_resume_argv != metadata.get("resume_argv"):
            changed = True
        if not changed:
            return snapshot

        metadata["request"] = request
        metadata["resume_argv"] = updated_resume_argv
        normalized = replace(snapshot, argv=updated_argv, metadata=metadata)
        self._write_snapshot(normalized)
        return normalized

    def _pid_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _pid_matches_snapshot(self, snapshot: DashboardJobSnapshot) -> bool:
        if snapshot.pid is None or not self._pid_is_alive(snapshot.pid):
            return False
        if snapshot.pid_token is None:
            return True
        return self._pid_token(snapshot.pid) == snapshot.pid_token

    def _pid_token(self, pid: int) -> str | None:
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            payload = stat_path.read_text(encoding="utf-8")
        except OSError:
            return None
        fields = payload.split()
        if len(fields) < 22:
            return None
        return fields[21]

    def _supervisor_argv(
        self, snapshot: DashboardJobSnapshot, command_argv: list[str]
    ) -> list[str]:
        argv = [
            self._python_executable,
            "-m",
            "dashboard.job_runner",
            "--job-root",
            str(snapshot.snapshot_path.parent),
        ]
        if snapshot.cwd is not None:
            argv.extend(["--cwd", str(snapshot.cwd)])
        argv.append("--")
        argv.extend(command_argv)
        return argv

    def _read_recent_output(self, log_path: Path) -> list[str]:
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return []
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[-self._max_recent_lines :]

    def _exit_code_path(self, snapshot: DashboardJobSnapshot) -> Path:
        return snapshot.snapshot_path.parent / "exit_code.txt"

    def _read_exit_code(self, snapshot: DashboardJobSnapshot) -> int | None:
        try:
            value = self._exit_code_path(snapshot).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _find_live_supervisor_pids(self, snapshot: DashboardJobSnapshot) -> list[int]:
        expected_job_root = str(snapshot.snapshot_path.parent.resolve())
        proc_root = Path("/proc")
        try:
            entries = list(proc_root.iterdir())
        except OSError:
            return []
        matches: list[int] = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                pid = int(entry.name)
            except ValueError:
                continue
            cmdline_path = entry / "cmdline"
            try:
                raw = cmdline_path.read_bytes()
            except OSError:
                continue
            if not raw:
                continue
            parts = [
                item.decode("utf-8", errors="ignore")
                for item in raw.split(b"\x00")
                if item
            ]
            if "dashboard.job_runner" not in parts:
                continue
            if "--job-root" not in parts:
                continue
            index = parts.index("--job-root")
            if index + 1 >= len(parts):
                continue
            try:
                actual_job_root = str(Path(parts[index + 1]).resolve())
            except OSError:
                actual_job_root = parts[index + 1]
            if actual_job_root != expected_job_root:
                continue
            if not self._pid_is_alive(pid):
                continue
            matches.append(pid)
        return sorted(set(matches))

    def _find_live_supervisor_pid(self, snapshot: DashboardJobSnapshot) -> int | None:
        pids = self._find_live_supervisor_pids(snapshot)
        if not pids:
            return None
        return self._preferred_supervisor_pid(snapshot, pids)

    def _preferred_supervisor_pid(
        self, snapshot: DashboardJobSnapshot, pids: list[int]
    ) -> int:
        if snapshot.pid in pids and self._pid_matches_snapshot(snapshot):
            return snapshot.pid
        return max(pids)

    def _terminate_supervisor_pid(self, pid: int, *, timeout: float = 5.0) -> None:
        if not self._pid_is_alive(pid):
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._pid_is_alive(pid):
                return
            time.sleep(0.05)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._pid_is_alive(pid):
                return
            time.sleep(0.05)

    def _append_dashboard_note(self, log_path: Path, note: str) -> None:
        line = f"[dashboard] {note}"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        except OSError:
            return

    def _reattach_live_supervisor(
        self, snapshot: DashboardJobSnapshot
    ) -> DashboardJobSnapshot | None:
        live_pids = self._find_live_supervisor_pids(snapshot)
        if not live_pids:
            return None
        live_pid = self._preferred_supervisor_pid(snapshot, live_pids)
        duplicate_pids = [pid for pid in live_pids if pid != live_pid]
        if duplicate_pids:
            note = (
                "Found multiple live supervisor trees for this job; "
                f"keeping pid {live_pid} and terminating duplicates {duplicate_pids}."
            )
            self._append_dashboard_note(snapshot.log_path, note)
            for pid in duplicate_pids:
                self._terminate_supervisor_pid(pid)
        recovered = self._replace_snapshot(
            snapshot,
            status="running",
            metadata=self._running_metadata(snapshot.metadata),
            finished_at=None,
            exit_code=None,
            pid=live_pid,
            pid_token=self._pid_token(live_pid),
            recent_output=self._read_recent_output(snapshot.log_path),
        )
        self._write_snapshot(recovered)
        return recovered

    def _recover_snapshot(self, snapshot: DashboardJobSnapshot) -> DashboardJobSnapshot:
        if snapshot.status == "failed" and self._is_queue_backed_queue_job(snapshot):
            wait_reason = self._queue_prerequisite_wait_reason(snapshot)
            if wait_reason is not None:
                note = (
                    "Queue-backed job was left failed while waiting on source artifacts; "
                    "returning it to the queue."
                )
                recovered = self._replace_snapshot(
                    snapshot,
                    status="queued",
                    metadata={
                        **self._terminal_metadata(snapshot.metadata, succeeded=False),
                        "queue_requested": True,
                        "launch_mode": "queue",
                        "queue_wait_reason": wait_reason,
                    },
                    started_at=None,
                    finished_at=None,
                    exit_code=None,
                    pid=None,
                    pid_token=None,
                    recent_output=[
                        *self._read_recent_output(snapshot.log_path)[-self._max_recent_lines + 1 :],
                        f"[dashboard] {note}",
                    ][-self._max_recent_lines :],
                )
                self._append_dashboard_note(snapshot.log_path, note)
                self._write_snapshot(recovered)
                return recovered
            retry_reason = self._classify_queue_recovery_failure(
                snapshot,
                exit_code=snapshot.exit_code,
            )
            if retry_reason is not None:
                note = (
                    "Queue-backed job failed with a retryable error before recovery; "
                    "returning it to the queue."
                )
                recovered = self._replace_snapshot(
                    snapshot,
                    status="queued",
                    metadata={
                        **self._terminal_metadata(snapshot.metadata, succeeded=False),
                        "queue_requested": True,
                        "launch_mode": "queue",
                    },
                    started_at=None,
                    finished_at=None,
                    exit_code=None,
                    pid=None,
                    pid_token=None,
                    recent_output=[
                        *self._read_recent_output(snapshot.log_path)[-self._max_recent_lines + 1 :],
                        f"[dashboard] {note}",
                    ][-self._max_recent_lines :],
                )
                self._append_dashboard_note(snapshot.log_path, note)
                self._write_snapshot(recovered)
                return recovered
        if snapshot.status == "queued":
            return snapshot
        if snapshot.status == "paused":
            return snapshot
        reattached = self._reattach_live_supervisor(snapshot)
        if reattached is not None:
            return reattached
        if snapshot.status != "running":
            return snapshot
        if self._pid_matches_snapshot(snapshot):
            return snapshot
        exit_code = self._read_exit_code(snapshot)
        if exit_code is not None:
            if exit_code != 0:
                retry_reason = self._classify_retryable_failure(snapshot, exit_code=exit_code)
                if retry_reason is not None:
                    note = (
                        "Dashboard job lost its supervisor process after a retryable failure; "
                        f"scheduling automatic resume ({retry_reason})."
                    )
                    recovered = self._paused_retry_snapshot(
                        snapshot,
                        exit_code=exit_code,
                        retry_reason=retry_reason,
                        note=note,
                    )
                    self._write_snapshot(recovered)
                    return recovered
            recovered_status: JobStatus = "succeeded" if exit_code == 0 else "failed"
            recovered = self._replace_snapshot(
                snapshot,
                status=recovered_status,
                metadata=self._terminal_metadata(
                    snapshot.metadata,
                    succeeded=exit_code == 0,
                ),
                finished_at=snapshot.finished_at or _utc_now(),
                exit_code=exit_code,
                pid=None,
                pid_token=None,
            )
            self._write_snapshot(recovered)
            return recovered
        note = (
            "Dashboard job lost its supervisor process while marked running; "
            "the server likely restarted before the job could be recovered, or the job was terminated externally."
        )
        if self._auto_resume_enabled(snapshot):
            recovered = self._paused_retry_snapshot(
                snapshot,
                exit_code=snapshot.exit_code if snapshot.exit_code is not None else -1,
                retry_reason="supervisor_lost",
                note=note + " Automatic resume has been scheduled.",
            )
            self._write_snapshot(recovered)
            return recovered
        recent_output = [
            *self._read_recent_output(snapshot.log_path)[-self._max_recent_lines + 1 :],
            f"[dashboard] {note}",
        ]
        try:
            snapshot.log_path.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[dashboard] {note}\n")
        except OSError:
            pass
        recovered = self._replace_snapshot(
            snapshot,
            status="failed",
            finished_at=snapshot.finished_at or _utc_now(),
            exit_code=snapshot.exit_code if snapshot.exit_code is not None else -1,
            pid=None,
            pid_token=None,
            recent_output=recent_output,
        )
        self._write_snapshot(recovered)
        return recovered

    def _maintenance_loop(self) -> None:
        while not self._maintenance_stop.wait(1.0):
            try:
                self._resume_due_auto_retry_jobs()
            except Exception:
                pass
            try:
                self._schedule_next_queued_job()
            except Exception:
                pass

    def _resume_due_auto_retry_jobs(self) -> None:
        due_job_ids: list[str] = []
        now = time.time()
        with self._lock:
            if self._queue_capacity_reached_locked():
                active_retry_ids = [
                    record.snapshot.job_id
                    for record in self._jobs.values()
                    if record.snapshot.status == "paused"
                    and record.snapshot.metadata.get("pause_reason")
                    == AUTO_RESUME_PAUSE_REASON
                    and self._auto_resume_due(record.snapshot.metadata, now=now)
                ]
                if not active_retry_ids:
                    return
            for record in self._jobs.values():
                snapshot = record.snapshot
                if snapshot.status != "paused":
                    continue
                if snapshot.metadata.get("pause_reason") != AUTO_RESUME_PAUSE_REASON:
                    continue
                if not self._auto_resume_due(snapshot.metadata, now=now):
                    continue
                due_job_ids.append(snapshot.job_id)
        for job_id in sorted(due_job_ids):
            try:
                self.resume(job_id)
            except (KeyError, ValueError):
                continue
            break

    def _initial_metadata(self, metadata: dict[str, Any], *, argv: list[str]) -> dict[str, Any]:
        normalized = dict(metadata)
        if self._metadata_supports_resume(normalized, argv=argv):
            normalized.setdefault("auto_resume_enabled", True)
            normalized.setdefault(
                "max_auto_resume_attempts",
                AUTO_RESUME_DEFAULT_MAX_ATTEMPTS,
            )
            normalized.setdefault(
                "auto_resume_backoff_seconds",
                AUTO_RESUME_DEFAULT_BACKOFF_SECONDS,
            )
            normalized.setdefault(
                "auto_resume_backoff_cap_seconds",
                AUTO_RESUME_DEFAULT_BACKOFF_CAP_SECONDS,
            )
        return normalized

    def _queue_order_value(self, snapshot: DashboardJobSnapshot) -> int:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        raw_value = metadata.get("queue_order")
        if isinstance(raw_value, bool):
            return sys.maxsize
        if isinstance(raw_value, int) and raw_value > 0:
            return raw_value
        if isinstance(raw_value, str):
            try:
                parsed = int(raw_value)
            except ValueError:
                return sys.maxsize
            if parsed > 0:
                return parsed
        return sys.maxsize

    def _queue_sort_key(self, snapshot: DashboardJobSnapshot) -> tuple[int, int, str, str]:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        action = str(metadata.get("action") or "")
        return (
            _QUEUE_DISPATCH_PRIORITY.get(action, 2),
            self._queue_order_value(snapshot),
            snapshot.created_at,
            snapshot.job_id,
        )

    def _next_queue_order_locked(self) -> int:
        max_value = 0
        for record in self._jobs.values():
            snapshot = record.snapshot
            metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
            if not bool(metadata.get("queue_requested")):
                continue
            if metadata.get("action") not in QUEUEABLE_JOB_ACTIONS:
                continue
            queue_order = self._queue_order_value(snapshot)
            if queue_order == sys.maxsize:
                continue
            max_value = max(max_value, queue_order)
        return max_value + 1

    def _ensure_queue_orders(self, records: dict[str, _JobRecord]) -> None:
        queue_records = sorted(
            (
                record
                for record in records.values()
                if bool(record.snapshot.metadata.get("queue_requested"))
                and record.snapshot.metadata.get("action") in QUEUEABLE_JOB_ACTIONS
                and self._queue_order_value(record.snapshot) == sys.maxsize
            ),
            key=lambda record: (
                record.snapshot.created_at,
                record.snapshot.job_id,
            ),
        )
        if not queue_records:
            return
        existing_max = 0
        for record in records.values():
            queue_order = self._queue_order_value(record.snapshot)
            if queue_order == sys.maxsize:
                continue
            existing_max = max(existing_max, queue_order)
        next_value = existing_max + 1 if existing_max > 0 else 1
        for record in queue_records:
            metadata = dict(record.snapshot.metadata)
            metadata["queue_order"] = next_value
            next_value += 1
            updated = self._replace_snapshot(record.snapshot, metadata=metadata)
            record.snapshot = updated
            self._write_snapshot(updated)

    def _transition_record_to_queued_locked(
        self,
        record: _JobRecord,
        *,
        note: str | None = None,
    ) -> DashboardJobSnapshot:
        metadata = dict(record.snapshot.metadata)
        queued_argv = record.snapshot.argv
        resume_argv = metadata.get("resume_argv")
        if isinstance(resume_argv, list) and resume_argv:
            queued_argv = [str(part) for part in resume_argv]
        metadata.pop("queue_wait_reason", None)
        metadata.pop("pause_reason", None)
        metadata.pop("auto_resume_due_at", None)
        metadata.pop("router_wait_started_at", None)
        metadata["queue_requested"] = True
        metadata["launch_mode"] = "queue"
        metadata["queue_order"] = self._next_queue_order_locked()
        recent_output = list(record.tail)
        if note:
            recent_output = self._append_dashboard_note_locked(record, note)
        updated = self._replace_snapshot(
            record.snapshot,
            status="queued",
            argv=queued_argv,
            metadata=metadata,
            started_at=None,
            finished_at=None,
            exit_code=None,
            recent_output=recent_output,
            pid=None,
            pid_token=None,
        )
        record.snapshot = updated
        record.process = None
        record.pause_requested = False
        record.requeue_requested = False
        record.tail = deque(updated.recent_output, maxlen=self._max_recent_lines)
        return updated

    def _metadata_supports_resume(
        self,
        metadata: dict[str, Any],
        *,
        argv: list[str],
    ) -> bool:
        if metadata.get("action") not in AUTO_RESUME_JOB_ACTIONS:
            return False
        if metadata.get("supports_pause", False):
            return True
        resume_argv = metadata.get("resume_argv")
        return isinstance(resume_argv, list) and bool(resume_argv or argv)

    def _auto_resume_enabled(self, snapshot: DashboardJobSnapshot) -> bool:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if not metadata.get("auto_resume_enabled", False):
            return False
        supports_resume = metadata.get("supports_pause", False) or (
            isinstance(metadata.get("resume_argv"), list)
            and bool(metadata.get("resume_argv"))
        )
        return bool(supports_resume)

    def _terminal_metadata(
        self,
        metadata: dict[str, Any],
        *,
        succeeded: bool,
    ) -> dict[str, Any]:
        updated = dict(metadata)
        updated.pop("pause_reason", None)
        updated.pop("auto_resume_due_at", None)
        updated.pop("auto_resume_last_failure_reason", None)
        updated.pop("router_wait_started_at", None)
        if succeeded:
            updated["auto_resume_attempt_count"] = 0
        return updated

    def _running_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        updated = dict(metadata)
        updated.pop("pause_reason", None)
        updated.pop("auto_resume_due_at", None)
        updated.pop("auto_resume_last_failure_reason", None)
        updated.pop("router_wait_started_at", None)
        updated["auto_resume_attempt_count"] = 0
        return updated

    def _paused_metadata(self, metadata: dict[str, Any], *, reason: str) -> dict[str, Any]:
        updated = dict(metadata)
        updated["pause_reason"] = reason
        updated.pop("auto_resume_due_at", None)
        if reason == ROUTER_WAIT_PAUSE_REASON:
            updated["router_wait_started_at"] = _utc_now()
        else:
            updated.pop("router_wait_started_at", None)
        return updated

    def _pause_note(self, reason: str) -> str:
        if reason == ROUTER_WAIT_PAUSE_REASON:
            return "Paused automatically because the LLM router has no active backend connections."
        return "Paused."

    def _append_dashboard_note_locked(
        self,
        record: _JobRecord,
        note: str,
    ) -> list[str]:
        line = f"[dashboard] {note}"
        record.tail.append(line)
        try:
            record.snapshot.log_path.parent.mkdir(parents=True, exist_ok=True)
            with record.snapshot.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        except OSError:
            pass
        return list(record.tail)

    def _classify_retryable_failure(
        self,
        snapshot: DashboardJobSnapshot,
        *,
        exit_code: int | None,
    ) -> str | None:
        if not self._auto_resume_enabled(snapshot):
            return None
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        attempts = _coerce_non_negative_int(metadata.get("auto_resume_attempt_count"))
        max_attempts = _coerce_positive_int(
            metadata.get("max_auto_resume_attempts"),
            default=AUTO_RESUME_DEFAULT_MAX_ATTEMPTS,
        )
        if attempts >= max_attempts:
            return None
        text = "\n".join(self._read_recent_output(snapshot.log_path)).lower()
        if any(pattern in text for pattern in TRANSIENT_LLM_FAILURE_PATTERNS):
            return "llm_transient_failure"
        if any(pattern in text for pattern in RESOURCE_FAILURE_PATTERNS):
            return "resource_pressure"
        if self._failure_snapshot_indicates_low_memory(snapshot):
            return "resource_pressure"
        if isinstance(exit_code, int) and exit_code in RESOURCE_FAILURE_EXIT_CODES:
            return "resource_pressure"
        return None

    def _classify_queue_recovery_failure(
        self,
        snapshot: DashboardJobSnapshot,
        *,
        exit_code: int | None,
    ) -> str | None:
        retry_reason = self._classify_retryable_failure(snapshot, exit_code=exit_code)
        if retry_reason is not None:
            return retry_reason
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        supports_resume = metadata.get("supports_pause", False) or (
            isinstance(metadata.get("resume_argv"), list)
            and bool(metadata.get("resume_argv"))
        )
        if not supports_resume:
            return None
        text = "\n".join(self._read_recent_output(snapshot.log_path)).lower()
        if any(pattern in text for pattern in TRANSIENT_LLM_FAILURE_PATTERNS):
            return "llm_transient_failure"
        if any(pattern in text for pattern in RESOURCE_FAILURE_PATTERNS):
            return "resource_pressure"
        if self._failure_snapshot_indicates_low_memory(snapshot):
            return "resource_pressure"
        if isinstance(exit_code, int) and exit_code in RESOURCE_FAILURE_EXIT_CODES:
            return "resource_pressure"
        return None

    def _failure_snapshot_indicates_low_memory(self, snapshot: DashboardJobSnapshot) -> bool:
        failure_snapshot_path = snapshot.snapshot_path.parent / "failed_resource_snapshot.json"
        try:
            payload = json.loads(failure_snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        system_memory = payload.get("system_memory")
        if isinstance(system_memory, dict):
            mem_available_kb = system_memory.get("mem_available_kb")
            mem_total_kb = system_memory.get("mem_total_kb")
            if isinstance(mem_available_kb, int) and mem_available_kb <= 2 * 1024 * 1024:
                return True
            if (
                isinstance(mem_available_kb, int)
                and isinstance(mem_total_kb, int)
                and mem_total_kb > 0
                and mem_available_kb / mem_total_kb <= 0.05
            ):
                return True
        return False

    def _transition_to_auto_resume_pause_locked(
        self,
        record: _JobRecord,
        *,
        exit_code: int | None,
        retry_reason: str,
    ) -> DashboardJobSnapshot:
        updated = self._paused_retry_snapshot(
            record.snapshot,
            exit_code=exit_code,
            retry_reason=retry_reason,
            note=(
                "Automatic resume scheduled after retryable failure "
                f"({retry_reason.replace('_', ' ')})."
            ),
        )
        record.snapshot = updated
        record.process = None
        record.pause_requested = False
        record.tail = deque(updated.recent_output, maxlen=self._max_recent_lines)
        return updated

    def _paused_retry_snapshot(
        self,
        snapshot: DashboardJobSnapshot,
        *,
        exit_code: int | None,
        retry_reason: str,
        note: str,
    ) -> DashboardJobSnapshot:
        metadata = dict(snapshot.metadata)
        attempts = _coerce_non_negative_int(metadata.get("auto_resume_attempt_count")) + 1
        delay_seconds = self._auto_resume_delay_seconds(metadata, attempt_count=attempts)
        due_at = _utc_from_epoch(time.time() + delay_seconds)
        metadata["auto_resume_attempt_count"] = attempts
        metadata["auto_resume_due_at"] = due_at
        metadata["auto_resume_last_failure_reason"] = retry_reason
        metadata["pause_reason"] = AUTO_RESUME_PAUSE_REASON
        recent_output = [
            *self._read_recent_output(snapshot.log_path)[-self._max_recent_lines + 1 :],
            f"[dashboard] {note} Next resume attempt at {due_at}.",
        ]
        try:
            snapshot.log_path.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[dashboard] {note} Next resume attempt at {due_at}.\n"
                )
        except OSError:
            pass
        return self._replace_snapshot(
            snapshot,
            status="paused",
            metadata=metadata,
            finished_at=_utc_now(),
            exit_code=exit_code,
            recent_output=recent_output[-self._max_recent_lines :],
            pid=None,
            pid_token=None,
        )

    def _auto_resume_delay_seconds(
        self,
        metadata: dict[str, Any],
        *,
        attempt_count: int,
    ) -> float:
        base = _coerce_positive_float(
            metadata.get("auto_resume_backoff_seconds"),
            default=AUTO_RESUME_DEFAULT_BACKOFF_SECONDS,
        )
        cap = _coerce_positive_float(
            metadata.get("auto_resume_backoff_cap_seconds"),
            default=AUTO_RESUME_DEFAULT_BACKOFF_CAP_SECONDS,
        )
        return min(base * max(1, attempt_count), cap)

    def _auto_resume_due(self, metadata: dict[str, Any], *, now: float) -> bool:
        due_at = metadata.get("auto_resume_due_at")
        due_epoch = _parse_utc_epoch(due_at)
        if due_epoch is None:
            return True
        return due_epoch <= now

    def _is_queue_backed_queue_job(self, snapshot: DashboardJobSnapshot) -> bool:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        return bool(metadata.get("queue_requested")) and metadata.get("action") in QUEUEABLE_JOB_ACTIONS

    def _occupies_launch_queue_capacity(self, snapshot: DashboardJobSnapshot) -> bool:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        return (
            snapshot.status in {"running", "paused"}
            and metadata.get("action") in QUEUEABLE_JOB_ACTIONS
        )

    def _is_pending_queue_job(self, snapshot: DashboardJobSnapshot) -> bool:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        return (
            snapshot.status == "queued"
            and bool(metadata.get("queue_requested"))
            and metadata.get("action") in QUEUEABLE_JOB_ACTIONS
        )

    def _active_queue_job_count_locked(self) -> int:
        return sum(
            self._occupies_launch_queue_capacity(record.snapshot)
            and not self._is_superseded_by_newer_succeeded_target_locked(record.snapshot)
            for record in self._jobs.values()
        )

    def _update_queue_wait_reason(
        self,
        job_id: str,
        *,
        reason: str | None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            snapshot = record.snapshot
            if snapshot.status != "queued":
                return
            metadata = dict(snapshot.metadata)
            previous_reason = metadata.get("queue_wait_reason")
            normalized_reason = str(reason).strip() if isinstance(reason, str) else None
            if normalized_reason:
                if previous_reason == normalized_reason:
                    return
                metadata["queue_wait_reason"] = normalized_reason
            else:
                if "queue_wait_reason" not in metadata:
                    return
                metadata.pop("queue_wait_reason", None)
            updated = self._replace_snapshot(snapshot, metadata=metadata)
            record.snapshot = updated
        self._write_snapshot(updated)

    def _resolve_auto_source_evaluation_results_path(
        self,
        *,
        source_run_dir: Path,
    ) -> Path | None:
        run_json_path = source_run_dir / "run.json"
        if run_json_path.is_file():
            try:
                payload = json.loads(run_json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                auto_run_id = payload.get("auto_evaluation_run_id")
                if isinstance(auto_run_id, str) and auto_run_id.strip():
                    candidate = source_run_dir / "deepphy_eval" / auto_run_id / "results.json"
                    if candidate.is_file():
                        return candidate
        deepphy_eval_root = source_run_dir / "deepphy_eval"
        if not deepphy_eval_root.is_dir():
            return None
        candidates = sorted(
            (path for path in deepphy_eval_root.glob("*/results.json") if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _queue_prerequisite_wait_reason(
        self,
        snapshot: DashboardJobSnapshot,
    ) -> str | None:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        if metadata.get("action") != "unsupervised-launch":
            return None
        request_payload = metadata.get("request")
        if not isinstance(request_payload, dict):
            return None
        source_run_dir_raw = request_payload.get("source_supervised_run_dir") or metadata.get(
            "source_supervised_run_dir"
        )
        source_library_raw = request_payload.get("source_pattern_library")
        source_evaluation_run_id = request_payload.get("source_evaluation_run_id") or metadata.get(
            "source_evaluation_run_id"
        )

        source_run_dir: Path | None = None
        source_library_path: Path | None = None
        if isinstance(source_library_raw, str) and source_library_raw.strip():
            source_library_path = Path(source_library_raw)
            source_run_dir = source_library_path.parent
        elif isinstance(source_run_dir_raw, str) and source_run_dir_raw.strip():
            source_run_dir = Path(source_run_dir_raw)
            source_library_path = source_run_dir / PATTERN_LIBRARY_FILENAME
        if source_run_dir is None or source_library_path is None:
            return None
        if not source_run_dir.is_dir():
            return f"Waiting for source supervised run directory: {source_run_dir}"
        normalized_source_run_dir = str(source_run_dir.resolve())
        with self._lock:
            for record in self._jobs.values():
                other = record.snapshot
                if other.job_id == snapshot.job_id:
                    continue
                other_metadata = (
                    other.metadata if isinstance(other.metadata, dict) else {}
                )
                if other_metadata.get("action") != "supervised-learn-and-evaluate":
                    continue
                other_run_dir = other_metadata.get("run_dir")
                if not isinstance(other_run_dir, str) or not other_run_dir.strip():
                    continue
                try:
                    normalized_other_run_dir = str(Path(other_run_dir).resolve())
                except OSError:
                    normalized_other_run_dir = other_run_dir.strip()
                if normalized_other_run_dir != normalized_source_run_dir:
                    continue
                if other.status != "succeeded":
                    return (
                        "Waiting for source supervised dashboard job to finish before "
                        "starting unsupervised learning."
                    )
        if not source_library_path.is_file():
            return f"Waiting for source supervised pattern library: {source_library_path}"
        if not isinstance(source_evaluation_run_id, str) or not source_evaluation_run_id.strip():
            return None
        if source_evaluation_run_id == "__auto_source_eval__":
            if self._resolve_auto_source_evaluation_results_path(source_run_dir=source_run_dir) is None:
                return (
                    "Waiting for source supervised evaluation results to be written before "
                    "starting unsupervised learning."
                )
            return None
        results_path = source_run_dir / "deepphy_eval" / source_evaluation_run_id / "results.json"
        if not results_path.is_file():
            return f"Waiting for source supervised evaluation results: {results_path}"
        return None

    def _queue_capacity_reached_locked(self) -> bool:
        if self._dispatching_job_ids:
            return True
        return self._active_queue_job_count_locked() >= self._max_concurrent_queue_jobs

    def _schedule_next_queued_job(self) -> DashboardJobSnapshot | None:
        started_snapshot: DashboardJobSnapshot | None = None
        while True:
            next_job_id: str | None = None
            with self._lock:
                if self._queue_capacity_reached_locked():
                    return started_snapshot
                queued_job_ids = [
                    snapshot.job_id
                    for snapshot in sorted(
                        (
                            record.snapshot
                            for record in self._jobs.values()
                            if self._is_pending_queue_job(record.snapshot)
                            and record.snapshot.job_id not in self._dispatching_job_ids
                        ),
                        key=self._queue_sort_key,
                    )
                ]
                if not queued_job_ids:
                    return started_snapshot
            for candidate_job_id in queued_job_ids:
                with self._lock:
                    record = self._jobs.get(candidate_job_id)
                    snapshot = record.snapshot if record is not None else None
                if snapshot is None or not self._is_pending_queue_job(snapshot):
                    continue
                wait_reason = self._queue_prerequisite_wait_reason(snapshot)
                if wait_reason is not None:
                    self._update_queue_wait_reason(candidate_job_id, reason=wait_reason)
                    continue
                self._update_queue_wait_reason(candidate_job_id, reason=None)
                with self._lock:
                    if self._queue_capacity_reached_locked():
                        return started_snapshot
                    record = self._jobs.get(candidate_job_id)
                    if record is None or not self._is_pending_queue_job(record.snapshot):
                        continue
                    self._dispatching_job_ids.add(candidate_job_id)
                    next_job_id = candidate_job_id
                break
            if next_job_id is None:
                return started_snapshot
            try:
                started_snapshot = self._start_job(next_job_id)
            finally:
                with self._lock:
                    self._dispatching_job_ids.discard(next_job_id)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except ValueError:
        return None


def _coerce_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value.strip()))
        except ValueError:
            return 0
    return 0


def _coerce_positive_int(value: Any, *, default: int) -> int:
    coerced = _coerce_non_negative_int(value)
    return coerced if coerced > 0 else default


def _coerce_positive_float(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        coerced = float(int(value))
    elif isinstance(value, (int, float)):
        coerced = float(value)
    elif isinstance(value, str):
        try:
            coerced = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    return coerced if coerced > 0.0 else default


def _replace_flag_value(argv: Any, flag: str, replacement: str) -> list[str]:
    values = [str(item) for item in list(argv or [])]
    if flag not in values:
        return values
    index = values.index(flag)
    if index + 1 >= len(values):
        return values
    updated = list(values)
    updated[index + 1] = replacement
    return updated
