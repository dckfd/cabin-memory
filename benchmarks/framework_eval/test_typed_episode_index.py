from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .typed_episode_index import (
    SQLiteTypedEpisodeIndex,
    TypedEpisodeConflictError,
    TypedEpisodeScope,
)


def _scope(*, user_id: str = "driver-a", task_id: str = "task-a") -> TypedEpisodeScope:
    return TypedEpisodeScope(
        conversation_id="conversation-a",
        team_id="cockpit-team",
        agent_id="vehicle-agent",
        user_id=user_id,
        task_id=task_id,
    )


def _episode(
    destination: str,
    *,
    source_ids: tuple[str, ...] = ("u1", "a1"),
    mentioned_at: str = "2026-08-26T08:00:00+08:00",
) -> dict:
    return {
        "schema_version": 1,
        "scene": "navigation",
        "intent": "navigation.set_destination",
        "state": "confirmed",
        "slots": {"destination": destination},
        "aliases": [],
        "source_ids": list(source_ids),
        "confidence": 0.995,
        "selection_actor": "assistant",
        "mentioned_at": mentioned_at,
        "request_text": "find a destination",
        "transitions": [],
    }


class SQLiteTypedEpisodeIndexTests(unittest.TestCase):
    def test_cold_restart_preserves_exact_namespace_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed-episodes.sqlite3"
            writer = SQLiteTypedEpisodeIndex(path)
            writer.upsert(
                _scope(),
                session_id="session-a",
                source_ids=("u1", "a1"),
                episode=_episode("虹桥机场"),
                now=100,
            )

            reader = SQLiteTypedEpisodeIndex(path)
            records = reader.list_active(_scope(), now=101)
            other_user = reader.list_active(
                _scope(user_id="driver-b"), now=101
            )
            other_task = reader.list_active(
                _scope(task_id="passenger-task"), now=101
            )
            integrity_ok = reader.integrity_check()

        self.assertTrue(integrity_ok)
        self.assertEqual(1, len(records))
        self.assertEqual("虹桥机场", records[0].episode["slots"]["destination"])
        self.assertEqual((), other_user)
        self.assertEqual((), other_task)

    def test_idempotent_replay_revision_and_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SQLiteTypedEpisodeIndex(Path(directory) / "episodes.db")
            first = index.upsert(
                _scope(),
                session_id="session-a",
                source_ids=("u1", "a1"),
                episode=_episode("虹桥机场"),
                ttl_seconds=60,
                now=100,
            )
            replay = index.upsert(
                _scope(),
                session_id="session-a",
                source_ids=("u1", "a1"),
                episode=_episode("虹桥机场"),
                ttl_seconds=60,
                now=120,
            )
            revised = index.upsert(
                _scope(),
                session_id="session-a",
                source_ids=("u1", "a1"),
                episode=_episode("浦东机场"),
                expected_revision=1,
                now=130,
            )

            with self.assertRaises(TypedEpisodeConflictError):
                index.upsert(
                    _scope(),
                    session_id="session-a",
                    source_ids=("u1", "a1"),
                    episode=_episode("上海站"),
                    expected_revision=1,
                    now=140,
                )

        self.assertEqual(1, first.revision)
        self.assertEqual(1, replay.revision)
        self.assertEqual(first.expires_at, replay.expires_at)
        self.assertEqual(2, revised.revision)
        self.assertEqual("浦东机场", revised.episode["slots"]["destination"])

    def test_ttl_invalidation_and_supersede_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SQLiteTypedEpisodeIndex(Path(directory) / "episodes.db")
            index.upsert(
                _scope(),
                session_id="old-route",
                source_ids=("u1", "a1"),
                episode=_episode("旧目的地"),
                ttl_seconds=10,
                now=100,
            )
            self.assertEqual(1, len(index.list_active(_scope(), now=109.9)))
            self.assertEqual((), index.list_active(_scope(), now=110))

            index.upsert(
                _scope(),
                session_id="replacement-route",
                source_ids=("u2", "a2"),
                episode=_episode(
                    "新目的地", source_ids=("u2", "a2"),
                    mentioned_at="2026-08-26T09:00:00+08:00",
                ),
                supersede_record_keys=("old-route",),
                now=120,
            )
            active = index.list_active(_scope(), now=121)
            invalidated = index.invalidate(
                _scope(), "replacement-route", reason="driver_cancelled", now=122
            )
            remaining = index.list_active(_scope(), now=123)

        self.assertEqual(("replacement-route",), tuple(
            record.record_key for record in active
        ))
        self.assertIsNotNone(invalidated)
        assert invalidated is not None
        self.assertFalse(invalidated.active)
        self.assertEqual(2, invalidated.revision)
        self.assertEqual("driver_cancelled", invalidated.invalidation_reason)
        self.assertEqual((), remaining)

    def test_concurrent_writers_commit_without_lost_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SQLiteTypedEpisodeIndex(
                Path(directory) / "episodes.db", busy_timeout_ms=10_000
            )

            def revise(number: int) -> None:
                index.upsert(
                    _scope(),
                    session_id="shared-session",
                    source_ids=("u1", "a1"),
                    episode=_episode(f"destination-{number}"),
                    now=100 + number,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(revise, range(8)))
            record = index.get(_scope(), "shared-session")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(8, record.revision)
        self.assertTrue(record.active)


if __name__ == "__main__":
    unittest.main()
