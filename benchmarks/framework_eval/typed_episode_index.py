from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class TypedEpisodeIndexError(RuntimeError):
    """The durable typed-episode index could not be read or updated safely."""


class TypedEpisodeConflictError(TypedEpisodeIndexError):
    """An optimistic episode update targeted a stale record revision."""


@dataclass(frozen=True)
class TypedEpisodeScope:
    """Exact tenant and memory-owner boundary for one episode record."""

    conversation_id: str
    team_id: str
    agent_id: str
    user_id: str
    task_id: str

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("typed episode scope requires conversation_id")
        if not self.agent_id:
            raise ValueError("typed episode scope requires agent_id")

    def values(self) -> tuple[str, str, str, str, str]:
        return (
            self.conversation_id,
            self.team_id,
            self.agent_id,
            self.user_id,
            self.task_id,
        )


@dataclass(frozen=True)
class TypedEpisodeRecord:
    scope: TypedEpisodeScope
    record_key: str
    session_id: str
    source_ids: tuple[str, ...]
    episode: dict[str, Any]
    revision: int
    active: bool
    created_at: float
    updated_at: float
    expires_at: float | None
    invalidated_at: float | None
    invalidation_reason: str


class SQLiteTypedEpisodeIndex:
    """Small durable episode index shared safely across adapter processes.

    Connections are intentionally short-lived. SQLite WAL plus an immediate
    write transaction gives ingestion workers atomic last-write semantics,
    while exact namespace predicates keep different drivers, assistants and
    benchmark conversations from sharing records.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path).resolve()
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS typed_episode_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)
                row = connection.execute(
                    "SELECT value FROM typed_episode_meta WHERE key = ?",
                    ("schema_version",),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO typed_episode_meta(key, value) VALUES (?, ?)",
                        ("schema_version", str(self.SCHEMA_VERSION)),
                    )
                elif int(row["value"]) != self.SCHEMA_VERSION:
                    raise TypedEpisodeIndexError(
                        "unsupported typed episode index schema: "
                        f"{row['value']} (expected {self.SCHEMA_VERSION})"
                    )
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS typed_episodes (
                        conversation_id TEXT NOT NULL,
                        team_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        record_key TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        scene TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        state TEXT NOT NULL,
                        mentioned_at TEXT NOT NULL,
                        source_ids_json TEXT NOT NULL,
                        episode_json TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        active INTEGER NOT NULL CHECK (active IN (0, 1)),
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        expires_at REAL,
                        invalidated_at REAL,
                        invalidation_reason TEXT NOT NULL DEFAULT '',
                        PRIMARY KEY (
                            conversation_id, team_id, agent_id, user_id,
                            task_id, record_key
                        )
                    )
                """)
                connection.execute("""
                    CREATE INDEX IF NOT EXISTS typed_episodes_active_scope
                    ON typed_episodes (
                        conversation_id, team_id, agent_id, user_id, task_id,
                        active, scene, mentioned_at
                    )
                """)
                connection.commit()
        except TypedEpisodeIndexError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise TypedEpisodeIndexError(
                f"cannot initialize typed episode index {self.path}: {exc}"
            ) from exc

    @staticmethod
    def _canonical_episode(
        episode: Mapping[str, Any], source_ids: Sequence[str]
    ) -> tuple[str, str, str, str, str, str]:
        if not isinstance(episode, Mapping):
            raise ValueError("typed episode payload must be an object")
        canonical_source_ids = tuple(
            dict.fromkeys(str(value) for value in source_ids if str(value))
        )
        if not canonical_source_ids:
            raise ValueError("typed episode record requires source_ids")
        episode_source_ids = tuple(
            str(value) for value in episode.get("source_ids") or [] if str(value)
        )
        if not episode_source_ids or not set(episode_source_ids).issubset(
            canonical_source_ids
        ):
            raise ValueError(
                "typed episode source_ids must be grounded in the source session"
            )
        scene = str(episode.get("scene") or "").strip().casefold()
        state = str(episode.get("state") or "").strip().casefold()
        if not scene or not state:
            raise ValueError("typed episode requires scene and state")
        episode_json = json.dumps(
            dict(episode), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        source_json = json.dumps(
            canonical_source_ids, ensure_ascii=False, separators=(",", ":")
        )
        payload_hash = hashlib.sha256(
            (source_json + "\x1f" + episode_json).encode("utf-8")
        ).hexdigest()
        return (
            source_json,
            episode_json,
            payload_hash,
            scene,
            str(episode.get("intent") or ""),
            state,
        )

    def upsert(
        self,
        scope: TypedEpisodeScope,
        *,
        session_id: str,
        source_ids: Sequence[str],
        episode: Mapping[str, Any],
        record_key: str = "",
        ttl_seconds: float = 0,
        active: bool = True,
        expected_revision: int | None = None,
        initial_revision: int = 1,
        supersede_record_keys: Sequence[str] = (),
        now: float | None = None,
    ) -> TypedEpisodeRecord:
        """Insert or atomically revise an episode within one exact scope.

        Replaying an identical payload is idempotent and does not extend its
        retention deadline. A changed payload increments ``revision``. Callers
        may pass ``expected_revision`` for compare-and-swap updates.
        """
        timestamp = time.time() if now is None else float(now)
        session_id = str(session_id)
        canonical_ids = tuple(
            dict.fromkeys(str(value) for value in source_ids if str(value))
        )
        key = str(record_key or session_id or "|".join(canonical_ids))
        if not key:
            raise ValueError("typed episode record requires a stable record key")
        if isinstance(initial_revision, bool) or int(initial_revision) < 1:
            raise ValueError("typed episode initial_revision must be positive")
        (
            source_json,
            episode_json,
            payload_hash,
            scene,
            intent,
            state,
        ) = self._canonical_episode(episode, canonical_ids)
        expires_at = (
            timestamp + float(ttl_seconds) if float(ttl_seconds) > 0 else None
        )
        scope_values = scope.values()
        try:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute("""
                    SELECT * FROM typed_episodes
                    WHERE conversation_id = ? AND team_id = ? AND agent_id = ?
                      AND user_id = ? AND task_id = ? AND record_key = ?
                """, (*scope_values, key)).fetchone()
                if existing is None:
                    if expected_revision not in {None, 0}:
                        raise TypedEpisodeConflictError(
                            f"typed episode {key} does not exist at revision "
                            f"{expected_revision}"
                        )
                    revision = int(initial_revision)
                    connection.execute("""
                        INSERT INTO typed_episodes (
                            conversation_id, team_id, agent_id, user_id,
                            task_id, record_key, session_id, scene, intent,
                            state, mentioned_at, source_ids_json, episode_json,
                            payload_hash, revision, active, created_at,
                            updated_at, expires_at, invalidated_at,
                            invalidation_reason
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, NULL, ''
                        )
                    """, (
                        *scope_values, key, session_id, scene, intent, state,
                        str(episode.get("mentioned_at") or ""), source_json,
                        episode_json, payload_hash, revision, int(active),
                        timestamp, timestamp, expires_at,
                    ))
                else:
                    current_revision = int(existing["revision"])
                    if (
                        expected_revision is not None
                        and expected_revision != current_revision
                    ):
                        raise TypedEpisodeConflictError(
                            f"typed episode {key} is at revision "
                            f"{current_revision}, expected {expected_revision}"
                        )
                    unchanged = (
                        existing["payload_hash"] == payload_hash
                        and existing["session_id"] == session_id
                        and bool(existing["active"]) == bool(active)
                    )
                    if unchanged:
                        revision = current_revision
                    else:
                        revision = current_revision + 1
                        connection.execute("""
                            UPDATE typed_episodes
                            SET session_id = ?, scene = ?, intent = ?, state = ?,
                                mentioned_at = ?, source_ids_json = ?,
                                episode_json = ?, payload_hash = ?, revision = ?,
                                active = ?, updated_at = ?, expires_at = ?,
                                invalidated_at = NULL,
                                invalidation_reason = ''
                            WHERE conversation_id = ? AND team_id = ?
                              AND agent_id = ? AND user_id = ? AND task_id = ?
                              AND record_key = ?
                        """, (
                            session_id, scene, intent, state,
                            str(episode.get("mentioned_at") or ""), source_json,
                            episode_json, payload_hash, revision, int(active),
                            timestamp, expires_at, *scope_values, key,
                        ))
                superseded = tuple(dict.fromkeys(
                    str(value) for value in supersede_record_keys
                    if str(value) and str(value) != key
                ))
                for superseded_key in superseded:
                    connection.execute("""
                        UPDATE typed_episodes
                        SET active = 0, revision = revision + 1,
                            updated_at = ?, invalidated_at = ?,
                            invalidation_reason = ?
                        WHERE conversation_id = ? AND team_id = ?
                          AND agent_id = ? AND user_id = ? AND task_id = ?
                          AND record_key = ? AND active = 1
                    """, (
                        timestamp, timestamp, f"superseded_by:{key}",
                        *scope_values, superseded_key,
                    ))
                row = connection.execute("""
                    SELECT * FROM typed_episodes
                    WHERE conversation_id = ? AND team_id = ? AND agent_id = ?
                      AND user_id = ? AND task_id = ? AND record_key = ?
                """, (*scope_values, key)).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        except (TypedEpisodeIndexError, TypedEpisodeConflictError):
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise TypedEpisodeIndexError(
                f"cannot upsert typed episode {key}: {exc}"
            ) from exc
        if row is None:
            raise TypedEpisodeIndexError(f"typed episode {key} vanished after upsert")
        return self._decode_row(row)

    def list_active(
        self,
        scope: TypedEpisodeScope,
        *,
        scene: str = "",
        now: float | None = None,
    ) -> tuple[TypedEpisodeRecord, ...]:
        timestamp = time.time() if now is None else float(now)
        params: list[Any] = [*scope.values(), timestamp]
        scene_clause = ""
        if scene:
            scene_clause = " AND scene = ?"
            params.append(str(scene).casefold())
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(f"""
                    SELECT * FROM typed_episodes
                    WHERE conversation_id = ? AND team_id = ? AND agent_id = ?
                      AND user_id = ? AND task_id = ? AND active = 1
                      AND (expires_at IS NULL OR expires_at > ?)
                      {scene_clause}
                    ORDER BY mentioned_at DESC, updated_at DESC, record_key ASC
                """, params).fetchall()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise TypedEpisodeIndexError(
                f"cannot query typed episode scope {scope}: {exc}"
            ) from exc
        return tuple(self._decode_row(row) for row in rows)

    def get(
        self, scope: TypedEpisodeScope, record_key: str
    ) -> TypedEpisodeRecord | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute("""
                    SELECT * FROM typed_episodes
                    WHERE conversation_id = ? AND team_id = ? AND agent_id = ?
                      AND user_id = ? AND task_id = ? AND record_key = ?
                """, (*scope.values(), str(record_key))).fetchone()
        except (OSError, sqlite3.Error) as exc:
            raise TypedEpisodeIndexError(
                f"cannot read typed episode {record_key}: {exc}"
            ) from exc
        return self._decode_row(row) if row is not None else None

    def invalidate(
        self,
        scope: TypedEpisodeScope,
        record_key: str,
        *,
        reason: str,
        expected_revision: int | None = None,
        now: float | None = None,
    ) -> TypedEpisodeRecord | None:
        timestamp = time.time() if now is None else float(now)
        key = str(record_key)
        if not reason.strip():
            raise ValueError("typed episode invalidation requires a reason")
        try:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("""
                    SELECT * FROM typed_episodes
                    WHERE conversation_id = ? AND team_id = ? AND agent_id = ?
                      AND user_id = ? AND task_id = ? AND record_key = ?
                """, (*scope.values(), key)).fetchone()
                if row is None:
                    connection.commit()
                    return None
                revision = int(row["revision"])
                if expected_revision is not None and expected_revision != revision:
                    raise TypedEpisodeConflictError(
                        f"typed episode {key} is at revision {revision}, "
                        f"expected {expected_revision}"
                    )
                if bool(row["active"]):
                    connection.execute("""
                        UPDATE typed_episodes
                        SET active = 0, revision = revision + 1,
                            updated_at = ?, invalidated_at = ?,
                            invalidation_reason = ?
                        WHERE conversation_id = ? AND team_id = ?
                          AND agent_id = ? AND user_id = ? AND task_id = ?
                          AND record_key = ?
                    """, (
                        timestamp, timestamp, reason.strip(),
                        *scope.values(), key,
                    ))
                updated = connection.execute("""
                    SELECT * FROM typed_episodes
                    WHERE conversation_id = ? AND team_id = ? AND agent_id = ?
                      AND user_id = ? AND task_id = ? AND record_key = ?
                """, (*scope.values(), key)).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        except (TypedEpisodeIndexError, TypedEpisodeConflictError):
            raise
        except (OSError, sqlite3.Error) as exc:
            raise TypedEpisodeIndexError(
                f"cannot invalidate typed episode {key}: {exc}"
            ) from exc
        return self._decode_row(updated) if updated is not None else None

    def integrity_check(self) -> bool:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute("PRAGMA quick_check").fetchone()
        except (OSError, sqlite3.Error) as exc:
            raise TypedEpisodeIndexError(
                f"cannot check typed episode index {self.path}: {exc}"
            ) from exc
        return bool(row and str(row[0]).casefold() == "ok")

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> TypedEpisodeRecord:
        try:
            source_ids = json.loads(row["source_ids_json"])
            episode = json.loads(row["episode_json"])
            if not isinstance(source_ids, list) or not isinstance(episode, dict):
                raise ValueError("record JSON has unexpected shape")
            scope = TypedEpisodeScope(
                conversation_id=str(row["conversation_id"]),
                team_id=str(row["team_id"]),
                agent_id=str(row["agent_id"]),
                user_id=str(row["user_id"]),
                task_id=str(row["task_id"]),
            )
            return TypedEpisodeRecord(
                scope=scope,
                record_key=str(row["record_key"]),
                session_id=str(row["session_id"]),
                source_ids=tuple(str(value) for value in source_ids),
                episode=episode,
                revision=int(row["revision"]),
                active=bool(row["active"]),
                created_at=float(row["created_at"]),
                updated_at=float(row["updated_at"]),
                expires_at=(
                    float(row["expires_at"])
                    if row["expires_at"] is not None else None
                ),
                invalidated_at=(
                    float(row["invalidated_at"])
                    if row["invalidated_at"] is not None else None
                ),
                invalidation_reason=str(row["invalidation_reason"] or ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TypedEpisodeIndexError(
                f"invalid typed episode index record: {exc}"
            ) from exc
