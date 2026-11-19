"""Checkpoint retention planner for training jobs.

Given a manifest of checkpoints (parent links, optimizer state, sizes,
upload status, digests, validation scores, storage tiers) and a retention
budget, produce a *plan* describing which checkpoint records must be kept
and which are safe to delete. The planner never touches storage itself;
it only emits the plan.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


UPLOAD_COMPLETE = "complete"
UPLOAD_UPLOADING = "uploading"
UPLOAD_PARTIAL = "partial"
UPLOAD_FAILED = "failed"

# Keep reasons emitted in the plan.
REASON_LATEST_USABLE = "latest_usable_on_recovery_chain"
REASON_ROLLBACK = "rollback_version"
REASON_BEST_METRIC = "best_validation_metric"
REASON_PERIODIC_ARCHIVE = "periodic_archive"
REASON_PINNED_ARCHIVE = "pinned_archive"
REASON_IN_PROGRESS = "upload_in_progress"
REASON_MULTI_TASK = "referenced_by_multiple_tasks"


@dataclass
class CheckpointRecord:
    """One checkpoint record in the manifest."""

    checkpoint_id: str
    task_id: str
    parent_id: Optional[str] = None
    created_at: float = 0.0
    size_bytes: int = 0
    optimizer_state_bytes: int = 0
    digest: Optional[str] = None
    upload_status: str = UPLOAD_COMPLETE
    validation_score: Optional[float] = None
    tier: str = "hot"
    is_archive: bool = False
    archive_until: Optional[float] = None
    task_refs: Tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return self.size_bytes + self.optimizer_state_bytes

    @property
    def is_usable(self) -> bool:
        """A checkpoint only counts as a usable version when its upload
        finished and its checksum digest is complete."""
        return self.upload_status == UPLOAD_COMPLETE and bool(self.digest)

    @property
    def referenced_task_ids(self) -> Tuple[str, ...]:
        return self.task_refs if self.task_refs else (self.task_id,)


@dataclass
class RetentionPolicy:
    """Knobs controlling the retention plan."""

    budget_bytes: int
    now: float = 0.0
    rollback_window: int = 1
    archive_interval_seconds: Optional[float] = None
    metric_mode: str = "max"  # "max" or "min"


@dataclass
class RetentionPlan:
    """The output of the planner. Pure data; performs no deletions."""

    keep: Dict[str, List[str]] = field(default_factory=dict)
    delete: List[str] = field(default_factory=list)
    orphans: List[str] = field(default_factory=list)
    freed_bytes: int = 0
    projected_bytes: int = 0
    unmet_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    def keep_reasons(self, checkpoint_id: str) -> List[str]:
        return list(self.keep.get(checkpoint_id, []))


def _content_key(record: CheckpointRecord) -> str:
    """Identity of the stored bytes. Records without a digest are treated
    as unique content (never deduplicated)."""
    if record.digest:
        return "digest:" + record.digest
    return "record:" + record.checkpoint_id


def _content_cost(records: List[CheckpointRecord]) -> int:
    """Storage cost of a set of records, sharing cost across records whose
    content has the same digest."""
    seen = {}
    for record in records:
        key = _content_key(record)
        if key not in seen:
            seen[key] = record.total_bytes
    return sum(seen.values())


def _find_orphans(records: Dict[str, CheckpointRecord]) -> List[str]:
    """Records whose declared parent is missing from the manifest."""
    orphans = []
    for record in records.values():
        if record.parent_id is not None and record.parent_id not in records:
            orphans.append(record.checkpoint_id)
    return sorted(orphans)


def _protect_latest_and_rollback(
    records: Dict[str, CheckpointRecord],
    policy: RetentionPolicy,
    keep: Dict[str, List[str]],
) -> None:
    """Protect the most recent usable checkpoint of every task's recovery
    chain plus a rollback window of earlier usable checkpoints."""
    by_task: Dict[str, List[CheckpointRecord]] = {}
    for record in records.values():
        by_task.setdefault(record.task_id, []).append(record)

    for task_records in by_task.values():
        usable = [r for r in task_records if r.is_usable]
        if not usable:
            continue
        latest = max(usable, key=lambda r: (r.created_at, r.checkpoint_id))
        keep.setdefault(latest.checkpoint_id, []).append(REASON_LATEST_USABLE)

        # Walk the parent chain from the latest usable checkpoint and
        # protect the rollback window of usable ancestors.
        remaining = policy.rollback_window
        cursor = latest
        while remaining > 0 and cursor.parent_id is not None:
            parent = records.get(cursor.parent_id)
            if parent is None:
                break
            if parent.is_usable:
                keep.setdefault(parent.checkpoint_id, []).append(REASON_ROLLBACK)
                remaining -= 1
            cursor = parent


def _protect_best_metric(
    records: Dict[str, CheckpointRecord],
    policy: RetentionPolicy,
    keep: Dict[str, List[str]],
) -> None:
    by_task: Dict[str, List[CheckpointRecord]] = {}
    for record in records.values():
        if record.is_usable and record.validation_score is not None:
            by_task.setdefault(record.task_id, []).append(record)

    reverse = policy.metric_mode != "min"
    for task_records in by_task.values():
        best = sorted(
            task_records,
            key=lambda r: (r.validation_score, r.created_at),
            reverse=reverse,
        )[0]
        keep.setdefault(best.checkpoint_id, []).append(REASON_BEST_METRIC)


def _protect_archives(
    records: Dict[str, CheckpointRecord],
    policy: RetentionPolicy,
    keep: Dict[str, List[str]],
) -> None:
    for record in records.values():
        if not record.is_usable:
            continue
        # Pinned archives only protect while unexpired.
        if record.is_archive:
            if record.archive_until is None or record.archive_until >= policy.now:
                keep.setdefault(record.checkpoint_id, []).append(REASON_PINNED_ARCHIVE)

    if policy.archive_interval_seconds:
        interval = policy.archive_interval_seconds
        buckets: Dict[Tuple[str, int], CheckpointRecord] = {}
        for record in records.values():
            if not record.is_usable:
                continue
            bucket = (record.task_id, int(record.created_at // interval))
            current = buckets.get(bucket)
            if current is None or (record.created_at, record.checkpoint_id) < (
                current.created_at,
                current.checkpoint_id,
            ):
                buckets[bucket] = record
        for record in buckets.values():
            keep.setdefault(record.checkpoint_id, []).append(REASON_PERIODIC_ARCHIVE)


def plan_retention(
    records: List[CheckpointRecord], policy: RetentionPolicy
) -> RetentionPlan:
    """Build a retention plan. Does not delete anything."""
    by_id = {r.checkpoint_id: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate checkpoint_id in manifest")

    plan = RetentionPlan()
    plan.orphans = _find_orphans(by_id)
    if plan.orphans:
        plan.notes.append(
            "orphaned checkpoints (missing parent): " + ", ".join(plan.orphans)
        )

    keep: Dict[str, List[str]] = {}

    # In-progress uploads must not be counted as usable, and must not be
    # deleted out from under an active writer.
    for record in records:
        if record.upload_status == UPLOAD_UPLOADING:
            keep.setdefault(record.checkpoint_id, []).append(REASON_IN_PROGRESS)

    # Records referenced by multiple tasks must not be dropped just because
    # their content duplicates another record.
    for record in records:
        if len(set(record.referenced_task_ids)) > 1:
            keep.setdefault(record.checkpoint_id, []).append(REASON_MULTI_TASK)

    _protect_latest_and_rollback(by_id, policy, keep)
    _protect_best_metric(by_id, policy, keep)
    _protect_archives(by_id, policy, keep)

    candidates = [r for r in records if r.checkpoint_id not in keep]
    # Delete oldest first; ties broken by id for determinism.
    candidates.sort(key=lambda r: (r.created_at, r.checkpoint_id))

    current_cost = _content_cost(records)
    remaining = list(records)
    freed = 0
    for record in candidates:
        if current_cost <= policy.budget_bytes:
            break
        remaining.remove(record)
        new_cost = _content_cost(remaining)
        freed += current_cost - new_cost
        current_cost = new_cost
        plan.delete.append(record.checkpoint_id)

    plan.keep = keep
    plan.freed_bytes = freed
    plan.projected_bytes = current_cost
    if current_cost > policy.budget_bytes:
        plan.unmet_bytes = current_cost - policy.budget_bytes
        plan.notes.append(
            "budget cannot be satisfied: {} bytes still over budget after "
            "deleting every eligible checkpoint".format(plan.unmet_bytes)
        )
    return plan
