import unittest

from checkpoint_planner import (
    REASON_BEST_METRIC,
    REASON_IN_PROGRESS,
    REASON_LATEST_USABLE,
    REASON_MULTI_TASK,
    REASON_PERIODIC_ARCHIVE,
    REASON_PINNED_ARCHIVE,
    REASON_ROLLBACK,
    UPLOAD_PARTIAL,
    UPLOAD_UPLOADING,
    CheckpointRecord,
    RetentionPolicy,
    plan_retention,
)


def ckpt(cid, task="t1", parent=None, ts=0.0, size=100, opt=0, digest=None,
         status="complete", score=None, **kwargs):
    return CheckpointRecord(
        checkpoint_id=cid,
        task_id=task,
        parent_id=parent,
        created_at=ts,
        size_bytes=size,
        optimizer_state_bytes=opt,
        digest=digest if digest is not None else "d-" + cid,
        upload_status=status,
        validation_score=score,
        **kwargs,
    )


class LinearChainTest(unittest.TestCase):
    def test_linear_chain_keeps_latest_and_rollback(self):
        records = [
            ckpt("c1", ts=1, size=100),
            ckpt("c2", parent="c1", ts=2, size=100),
            ckpt("c3", parent="c2", ts=3, size=100),
            ckpt("c4", parent="c3", ts=4, size=100),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=200, rollback_window=1))
        self.assertIn(REASON_LATEST_USABLE, plan.keep_reasons("c4"))
        self.assertIn(REASON_ROLLBACK, plan.keep_reasons("c3"))
        self.assertEqual(sorted(plan.delete), ["c1", "c2"])
        self.assertEqual(plan.freed_bytes, 200)
        self.assertEqual(plan.projected_bytes, 200)
        self.assertEqual(plan.unmet_bytes, 0)

    def test_orphan_reported_when_parent_missing(self):
        records = [
            ckpt("c2", parent="gone", ts=2, size=100),
            ckpt("c3", parent="c2", ts=3, size=100),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=1000))
        self.assertEqual(plan.orphans, ["c2"])
        self.assertTrue(any("c2" in note for note in plan.notes))


class ForkTest(unittest.TestCase):
    def test_forked_chains_keep_both_heads(self):
        records = [
            ckpt("base", task="ta", ts=1, size=100),
            ckpt("a1", task="ta", parent="base", ts=2, size=100),
            ckpt("a2", task="ta", parent="a1", ts=3, size=100),
            ckpt("b1", task="tb", parent="base", ts=2, size=100),
            ckpt("b2", task="tb", parent="b1", ts=3, size=100),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=10 ** 9, rollback_window=1))
        self.assertIn(REASON_LATEST_USABLE, plan.keep_reasons("a2"))
        self.assertIn(REASON_LATEST_USABLE, plan.keep_reasons("b2"))
        self.assertIn(REASON_ROLLBACK, plan.keep_reasons("a1"))
        self.assertIn(REASON_ROLLBACK, plan.keep_reasons("b1"))
        # Shared ancestor is on both recovery chains' rollback path only via
        # a1/b1's parents; with window=1 it is not protected.
        self.assertNotIn("base", plan.keep)

    def test_shared_ancestor_protected_with_wider_window(self):
        records = [
            ckpt("base", ts=1, size=100),
            ckpt("a1", task="ta", parent="base", ts=2, size=100),
            ckpt("a2", task="ta", parent="a1", ts=3, size=100),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=0, rollback_window=2))
        self.assertIn(REASON_ROLLBACK, plan.keep_reasons("base"))
        self.assertEqual(plan.delete, [])


class DuplicateDigestTest(unittest.TestCase):
    def test_shared_digest_shares_cost(self):
        records = [
            ckpt("c1", ts=1, size=100, digest="same"),
            ckpt("c2", parent="c1", ts=2, size=100, digest="same"),
            ckpt("c3", parent="c2", ts=3, size=100, digest="unique"),
        ]
        # Unique content: "same" (100) + "unique" (100) = 200 total.
        plan = plan_retention(records, RetentionPolicy(budget_bytes=100, rollback_window=0))
        # c3 kept (latest). Deleting c1 frees 0 because c2 still holds "same".
        self.assertEqual(plan.delete[0], "c1")
        self.assertEqual(plan.freed_bytes, 100)
        self.assertEqual(plan.projected_bytes, 100)

    def test_multi_task_reference_not_deleted_for_identical_content(self):
        records = [
            ckpt("shared", task="ta", ts=1, size=100, digest="same",
                 task_refs=("ta", "tb")),
            ckpt("dup", task="tb", ts=2, size=100, digest="same"),
            ckpt("head_a", task="ta", parent="shared", ts=3, size=100),
            ckpt("head_b", task="tb", parent="dup", ts=3, size=100),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=0, rollback_window=0))
        self.assertIn(REASON_MULTI_TASK, plan.keep_reasons("shared"))
        self.assertNotIn("shared", plan.delete)
        # The plain duplicate record is still eligible for deletion.
        self.assertIn("dup", plan.delete)
        # Freeing "dup" releases nothing while "shared" still references it.
        self.assertEqual(plan.freed_bytes, 0)


class HalfUploadTest(unittest.TestCase):
    def test_uploading_checkpoint_not_usable_and_not_deleted(self):
        records = [
            ckpt("c1", ts=1, size=100),
            ckpt("c2", parent="c1", ts=2, size=100, status=UPLOAD_UPLOADING),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=0, rollback_window=1))
        self.assertIn(REASON_IN_PROGRESS, plan.keep_reasons("c2"))
        # c1 is the most recent *usable* checkpoint and must be kept.
        self.assertIn(REASON_LATEST_USABLE, plan.keep_reasons("c1"))
        self.assertEqual(plan.delete, [])
        self.assertEqual(plan.unmet_bytes, 200)

    def test_incomplete_digest_not_counted_as_usable(self):
        records = [
            ckpt("c1", ts=1, size=100),
            ckpt("c2", parent="c1", ts=2, size=100, digest=None,
                 status=UPLOAD_PARTIAL),
        ]
        records[1].digest = None
        plan = plan_retention(records, RetentionPolicy(budget_bytes=100, rollback_window=1))
        self.assertIn(REASON_LATEST_USABLE, plan.keep_reasons("c1"))
        # Partial upload with no digest is not usable and may be cleaned up.
        self.assertIn("c2", plan.delete)
        self.assertEqual(plan.freed_bytes, 100)


class BestMetricTest(unittest.TestCase):
    def test_best_metric_protected_beyond_rollback_window(self):
        records = [
            ckpt("c1", ts=1, size=100, score=0.9),
            ckpt("c2", parent="c1", ts=2, size=100, score=0.7),
            ckpt("c3", parent="c2", ts=3, size=100, score=0.8),
            ckpt("c4", parent="c3", ts=4, size=100, score=0.75),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=0, rollback_window=0))
        self.assertIn(REASON_BEST_METRIC, plan.keep_reasons("c1"))
        self.assertNotIn("c1", plan.delete)
        self.assertEqual(sorted(plan.delete), ["c2", "c3"])

    def test_best_metric_min_mode(self):
        records = [
            ckpt("c1", ts=1, size=100, score=0.9),
            ckpt("c2", parent="c1", ts=2, size=100, score=0.1),
            ckpt("c3", parent="c2", ts=3, size=100, score=0.5),
        ]
        plan = plan_retention(
            records, RetentionPolicy(budget_bytes=10 ** 9, rollback_window=0,
                                     metric_mode="min")
        )
        self.assertIn(REASON_BEST_METRIC, plan.keep_reasons("c2"))


class ArchiveTest(unittest.TestCase):
    def test_unexpired_pinned_archive_kept(self):
        records = [
            ckpt("old", ts=1, size=100, is_archive=True, archive_until=1000),
            ckpt("head", parent="old", ts=2, size=100),
        ]
        plan = plan_retention(
            records, RetentionPolicy(budget_bytes=0, now=500, rollback_window=0)
        )
        self.assertIn(REASON_PINNED_ARCHIVE, plan.keep_reasons("old"))

    def test_expired_archive_becomes_deletable(self):
        records = [
            ckpt("old", ts=1, size=100, is_archive=True, archive_until=100),
            ckpt("head", parent="old", ts=2, size=100),
        ]
        plan = plan_retention(
            records, RetentionPolicy(budget_bytes=0, now=500, rollback_window=0)
        )
        self.assertNotIn("old", plan.keep)
        self.assertIn("old", plan.delete)
        self.assertEqual(plan.freed_bytes, 100)

    def test_periodic_archive_keeps_first_of_each_interval(self):
        records = [
            ckpt("c1", ts=0, size=100),
            ckpt("c2", parent="c1", ts=50, size=100),
            ckpt("c3", parent="c2", ts=150, size=100),
            ckpt("c4", parent="c3", ts=250, size=100),
        ]
        plan = plan_retention(
            records,
            RetentionPolicy(budget_bytes=10 ** 9, rollback_window=0,
                            archive_interval_seconds=100),
        )
        for cid in ("c1", "c3", "c4"):
            self.assertIn(cid, plan.keep)
        self.assertIn(REASON_PERIODIC_ARCHIVE, plan.keep_reasons("c1"))
        self.assertIn(REASON_PERIODIC_ARCHIVE, plan.keep_reasons("c3"))
        # c2 falls in the same interval bucket as c1 and is not protected.
        self.assertNotIn("c2", plan.keep)


class BudgetTest(unittest.TestCase):
    def test_insufficient_budget_reports_unmet(self):
        records = [
            ckpt("c1", ts=1, size=100, opt=50),
            ckpt("c2", parent="c1", ts=2, size=100, opt=50),
            ckpt("c3", parent="c2", ts=3, size=100, opt=50),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=100, rollback_window=1))
        # c2 + c3 are protected (rollback + latest): 300 bytes remain.
        self.assertEqual(sorted(plan.delete), ["c1"])
        self.assertEqual(plan.freed_bytes, 150)
        self.assertEqual(plan.projected_bytes, 300)
        self.assertEqual(plan.unmet_bytes, 200)
        self.assertTrue(any("budget" in note for note in plan.notes))

    def test_optimizer_state_counts_toward_cost(self):
        records = [
            ckpt("c1", ts=1, size=100, opt=900),
            ckpt("c2", parent="c1", ts=2, size=100, opt=900),
        ]
        plan = plan_retention(records, RetentionPolicy(budget_bytes=1500, rollback_window=0))
        self.assertEqual(plan.delete, ["c1"])
        self.assertEqual(plan.freed_bytes, 1000)
        self.assertEqual(plan.projected_bytes, 1000)


if __name__ == "__main__":
    unittest.main()
