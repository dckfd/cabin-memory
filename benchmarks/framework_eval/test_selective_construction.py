from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from .adapters.construction_policy import classify_session, classify_session_v2
from .adapters.tencentdb_http import TencentDBHTTPAdapter, _v3_source_ids
from .schema import Conversation, Message, Session
from .typed_episode_index import SQLiteTypedEpisodeIndex, TypedEpisodeScope


def _session(
    user: str,
    assistant: str = "",
    *,
    intent: str = "",
    domain: str = "",
    session_id: str = "session",
) -> Session:
    messages = [Message(
        f"{session_id}-u1", "user", user, speaker="Driver"
    )]
    if assistant:
        messages.append(Message(
            f"{session_id}-a1", "assistant", assistant,
            speaker="Car Assistant"
        ))
    return Session(
        session_id, "2026-01-01T08:00:00Z", tuple(messages),
        {"source_intent": intent, "source_domain": domain},
    )


class SelectiveConstructionTests(unittest.TestCase):
    def test_policy_selects_profile_event_and_correction(self):
        profile = classify_session(_session("Always avoid toll roads."))
        event = classify_session(_session(
            "Schedule a meeting tomorrow at 3 pm.",
            "The meeting has been scheduled.",
            intent="calendar_create",
            domain="calendar",
        ))
        correction = classify_session(_session(
            "Actually, change that destination to the airport."
        ))
        self.assertEqual(("profile", True), (
            profile.memory_type, profile.extract_l1
        ))
        self.assertEqual(("event", True), (event.memory_type, event.extract_l1))
        self.assertEqual("user_correction", correction.reason)

    def test_policy_keeps_queries_and_acknowledgements_out_of_l1(self):
        query = classify_session(_session(
            "What is the weather in Shanghai?",
            "It will be sunny.",
            intent="weather_query",
            domain="weather",
        ))
        acknowledgement = classify_session(_session("Thanks!"))
        self.assertEqual(("record", False), (
            query.memory_type, query.extract_l1
        ))
        self.assertEqual(("transient", False), (
            acknowledgement.memory_type, acknowledgement.extract_l1
        ))

    def test_episode_v2_distinguishes_durable_and_transient_commands(self):
        calendar = classify_session_v2(_session(
            "Schedule lunch tomorrow at noon.",
            "What title should I use?",
            intent="calendar_create",
            domain="calendar",
        ))
        playback = classify_session_v2(_session(
            "Play some jazz.",
            "What artist should I use?",
            intent="play_music",
            domain="audio",
        ))
        cancelled = classify_session_v2(_session(
            "Cancel my appointment tomorrow.",
            "The appointment has been cancelled.",
            intent="calendar_remove",
            domain="calendar",
        ))

        self.assertEqual(
            ("event", True, "pending", "pending", "calendar"),
            (
                calendar.memory_type,
                calendar.extract_l1,
                calendar.write_action,
                calendar.lifecycle,
                calendar.scene,
            ),
        )
        self.assertEqual(
            ("record", False, "retain", "media"),
            (
                playback.memory_type,
                playback.extract_l1,
                playback.write_action,
                playback.scene,
            ),
        )
        self.assertEqual(
            ("event", True, "expire", "confirmed"),
            (
                cancelled.memory_type,
                cancelled.extract_l1,
                cancelled.write_action,
                cancelled.lifecycle,
            ),
        )

    def test_episode_v2_keeps_general_profile_and_event_coverage(self):
        identity = classify_session_v2(_session(
            "I live in Shenzhen and work for a robotics company."
        ))
        experience = classify_session_v2(_session(
            "I visited Hangzhou last weekend."
        ))
        self.assertEqual(("profile", True, "update"), (
            identity.memory_type, identity.extract_l1, identity.write_action
        ))
        self.assertEqual(("event", True, "add"), (
            experience.memory_type, experience.extract_l1,
            experience.write_action,
        ))

    def _adapter(
        self, directory: str, **environment: str
    ) -> TencentDBHTTPAdapter:
        manifest = {
            "team_id": "team",
            "user_id": "user",
            "conversations": {
                "conv": {"agent_id": "agent", "task_id": "task"},
            },
        }
        manifest_path = Path(directory) / "isolation.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        trace_path = Path(directory) / "construction.jsonl"
        patcher = mock.patch.dict("os.environ", {
            "TDAI_EVAL_ISOLATION_MAP": str(manifest_path),
            "TDAI_HTTP_API_VERSION": "v3",
            "TDAI_EVAL_PERSPECTIVE_MODE": "single",
            "TDAI_EVAL_L1_WRITE_POLICY": "cockpit_selective_v1",
            "TDAI_EVAL_CONSTRUCTION_TRACE": str(trace_path),
            "TDAI_EVAL_MEMORY_LAYERS": "L0,L1",
            "TDAI_EVAL_L23_SCHEDULE": "disabled",
            "TDAI_EVAL_L23_READINESS_MODE": "required",
            **environment,
        }, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        return TencentDBHTTPAdapter("http://unused")

    def test_adapter_suppresses_pipeline_trigger_but_keeps_l0_content(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            payloads = []

            def fake_post(endpoint, body):
                self.assertEqual("/v3/conversation/add", endpoint)
                payloads.append(body)
                return {"code": 0, "data": {}}

            session = _session(
                "What is the weather in Shanghai?",
                "It will be sunny.",
                intent="weather_query",
                domain="weather",
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                adapter.ingest_session("conv", session)
            trace = json.loads(
                (Path(directory) / "construction.jsonl").read_text(encoding="utf-8")
            )
            metrics = adapter.construction_metrics()

        self.assertEqual(["assistant", "assistant"], [
            value["role"] for value in payloads[0]["messages"]
        ])
        self.assertIn("Driver: What is the weather", payloads[0]["messages"][0]["content"])
        self.assertTrue(trace["l1_trigger_suppressed"])
        self.assertEqual(0, trace["pipeline_user_rounds"])
        self.assertEqual(1, metrics["suppressed_from_l1"])
        self.assertEqual(2, metrics["l0_messages_written"])

    def test_adapter_demotes_tool_transport_role_but_keeps_tool_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            payloads = []
            session = Session(
                "climate-1",
                "2026-08-26T09:00:00+08:00",
                (
                    Message(
                        "climate-u1", "user", "把后排左侧调到 22 度。",
                        speaker="Driver",
                    ),
                    Message(
                        "climate-t1", "tool", "rear_left_temperature=22; status=success",
                        speaker="vehicle_control",
                    ),
                ),
                {"source_intent": "climate_set", "source_domain": "climate"},
            )

            with mock.patch.object(
                adapter,
                "_post",
                side_effect=lambda _endpoint, body: (
                    payloads.append(body) or {"code": 0, "data": {}}
                ),
            ):
                adapter.ingest_session("conv", session)

        tool_row = next(
            row for row in payloads[0]["messages"]
            if "climate-t1" in row["content"]
        )
        self.assertEqual("assistant", tool_row["role"])
        self.assertIn("[source_role=tool]", tool_row["content"])
        self.assertIn("status=success", tool_row["content"])

    def test_http_error_includes_response_body(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            failure = urllib.error.HTTPError(
                url="http://unused/v3/conversation/add",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"invalid conversation role"}'),
            )

            with mock.patch("urllib.request.urlopen", side_effect=failure):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"/v3/conversation/add failed: HTTP 400: .*invalid conversation role",
                ):
                    adapter._post("/v3/conversation/add", {"messages": []})

    def test_adapter_preserves_user_trigger_for_selected_event(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(directory)
            payloads = []

            def fake_post(_endpoint, body):
                payloads.append(body)
                return {"code": 0, "data": {}}

            session = _session(
                "Schedule lunch tomorrow at noon.",
                "Lunch has been scheduled.",
                intent="calendar_create",
                domain="calendar",
            )
            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                adapter.ingest_session("conv", session)
            metrics = adapter.construction_metrics()

        self.assertEqual("user", payloads[0]["messages"][0]["role"])
        self.assertEqual(1, metrics["selected_for_l1"])
        self.assertEqual(1.0, metrics["selection_rate"])

    def test_conversation_batch_coalesces_l1_and_restores_source_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_BATCH_MAX_MESSAGES="128",
            )
            payloads = []

            def capture_post(endpoint, body):
                self.assertEqual("/v3/conversation/add", endpoint)
                payloads.append(body)
                return {"code": 0, "data": {}}

            conversation = Conversation("conv", (
                _session(
                    "Schedule lunch tomorrow at noon.",
                    "Lunch has been scheduled.",
                    domain="calendar",
                    session_id="selected-1",
                ),
                _session(
                    "Remind me to call Alex at 3 pm.",
                    "The reminder is set.",
                    domain="calendar",
                    session_id="selected-2",
                ),
                _session(
                    "What is the weather?", "It is sunny.",
                    domain="weather", session_id="record-1",
                ),
            ))
            with mock.patch.object(adapter, "_post", side_effect=capture_post):
                adapter.ingest(conversation)

            trace_rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "construction.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            metrics = adapter.construction_metrics()

            # The record-only source session is written immediately. Both
            # selected sessions are flushed through one shared L1 cursor.
            self.assertEqual(2, len(payloads))
            batch = next(
                payload for payload in payloads
                if "selective-l1-conversation-batch" in payload["session_id"]
            )
            self.assertEqual(4, len(batch["messages"]))
            self.assertEqual(1, metrics["l1_batches_written"])
            selected_transports = {
                row["transport_session_id"] for row in trace_rows
                if row["decision"]["extract_l1"]
            }
            self.assertEqual({batch["session_id"]}, selected_transports)

            backend_rows = []
            for payload_index, payload in enumerate(payloads):
                for message_index, message in enumerate(payload["messages"]):
                    backend_rows.append({
                        "id": f"b-{payload_index}-{message_index}",
                        "session_id": payload["session_id"],
                        "role": message["role"],
                        "content": message["content"],
                        "timestamp": f"2026-01-01T08:00:{len(backend_rows):02d}Z",
                    })
            duplicate_source_ids = tuple(_v3_source_ids(
                backend_rows[0]["content"]
            ))
            backend_rows.append({
                **backend_rows[0],
                "id": "b-replay",
                "timestamp": "2026-01-01T09:00:00Z",
            })

            with mock.patch.object(adapter, "_post", return_value={
                "code": 0,
                "data": {"messages": backend_rows, "total": len(backend_rows)},
            }):
                history = adapter._load_v3_l0_history("conv")

        self.assertEqual(
            {"selected-1", "selected-2", "record-1"},
            set(history.sessions),
        )
        self.assertEqual(
            {2}, {len(messages) for messages in history.sessions.values()}
        )
        self.assertEqual(
            history.by_backend_id["b-replay"],
            history.by_source_id[duplicate_source_ids[0]],
        )

    def test_conversation_batch_coalesces_clean_l0_transport_only(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L0_BATCH_MODE="conversation",
                TDAI_EVAL_L0_BATCH_MAX_MESSAGES="128",
            )
            payloads = []

            def capture_post(endpoint, body):
                self.assertEqual("/v3/conversation/add", endpoint)
                payloads.append(body)
                return {"code": 0, "data": {}}

            conversation = Conversation("conv", (
                _session(
                    "What is the weather in Shanghai?", "It is sunny.",
                    domain="weather", session_id="record-1",
                ),
                _session(
                    "Play some jazz.", "Playing jazz.",
                    domain="audio", session_id="record-2",
                ),
            ))
            with mock.patch.object(adapter, "_post", side_effect=capture_post):
                adapter.ingest(conversation)

            trace_rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "construction.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            metrics = adapter.construction_metrics()
            self.assertEqual(1, len(payloads))
            self.assertIn(
                "lossless-l0-conversation-batch", payloads[0]["session_id"]
            )
            self.assertEqual(4, len(payloads[0]["messages"]))
            self.assertEqual(
                {"assistant"},
                {message["role"] for message in payloads[0]["messages"]},
            )
            self.assertTrue(all(row["l0_batched"] for row in trace_rows))
            self.assertEqual(1, metrics["l0_batches_written"])
            self.assertEqual(2, metrics["l0_batched_sessions"])
            self.assertEqual(1, metrics["l0_transport_requests_saved"])

            backend_rows = [{
                "id": f"b-{index}",
                "session_id": payloads[0]["session_id"],
                "role": message["role"],
                "content": message["content"],
                "timestamp": f"2026-01-01T08:00:0{index}Z",
            } for index, message in enumerate(payloads[0]["messages"])]
            reader = self._adapter(
                directory,
                TDAI_EVAL_L0_BATCH_MODE="conversation",
            )
            with mock.patch.object(reader, "_post", return_value={
                "code": 0,
                "data": {"messages": backend_rows, "total": len(backend_rows)},
            }):
                history = reader._load_v3_l0_history("conv")

        self.assertEqual({"record-1", "record-2"}, set(history.sessions))
        self.assertEqual({2}, {len(rows) for rows in history.sessions.values()})

    def test_l0_batch_size_is_clamped_to_memorycore_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L0_BATCH_MODE="conversation",
                TDAI_EVAL_L0_BATCH_MAX_MESSAGES="128",
            )

        self.assertEqual(100, adapter.l0_batch_max_messages)

    def test_clean_l0_sessions_can_flush_concurrently_without_merging(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L0_BATCH_MODE="session",
                TDAI_EVAL_L0_FLUSH_CONCURRENCY="2",
            )
            payloads = []
            with mock.patch.object(
                adapter,
                "_post",
                side_effect=lambda _endpoint, body: (
                    payloads.append(body) or {"code": 0, "data": {}}
                ),
            ):
                adapter.ingest(Conversation("conv", (
                    _session(
                        "What is the weather?", "It is sunny.",
                        domain="weather", session_id="record-1",
                    ),
                    _session(
                        "Play jazz.", "Playing jazz.",
                        domain="audio", session_id="record-2",
                    ),
                )))
            trace_rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "construction.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            metrics = adapter.construction_metrics()

        self.assertEqual(2, len(payloads))
        self.assertEqual(2, len({item["session_id"] for item in payloads}))
        self.assertTrue(all(row["l0_buffered"] for row in trace_rows))
        self.assertTrue(all(not row["l0_batched"] for row in trace_rows))
        self.assertEqual(2, metrics["l0_concurrent_requests"])
        self.assertEqual(2, metrics["l0_flush_concurrency"])

    def test_conversation_batch_compacts_each_selected_source_session(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_BATCH_MAX_MESSAGES="128",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
            )
            payloads = []

            def capture_post(endpoint, body):
                self.assertEqual("/v3/conversation/add", endpoint)
                payloads.append(body)
                return {"code": 0, "data": {}}

            conversation = Conversation("conv", (
                _session(
                    "Schedule lunch tomorrow at noon.",
                    "Lunch has been scheduled.",
                    domain="calendar",
                    session_id="selected-1",
                ),
                _session(
                    "Remind me to call Alex at 3 pm.",
                    "The reminder is set.",
                    domain="calendar",
                    session_id="selected-2",
                ),
            ))
            with mock.patch.object(adapter, "_post", side_effect=capture_post):
                adapter.ingest(conversation)

            batch = payloads[0]
            trace_rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "construction.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            metrics = adapter.construction_metrics()

            self.assertEqual(2, len(batch["messages"]))
            self.assertEqual(["user", "user"], [
                message["role"] for message in batch["messages"]
            ])
            self.assertIn("[selected-1-u1]", batch["messages"][0]["content"])
            self.assertIn("[selected-1-a1]", batch["messages"][0]["content"])
            self.assertEqual([1, 1], [
                row["pipeline_user_rounds"] for row in trace_rows
            ])
            self.assertEqual([2, 2], [
                len(row["source_ids"]) for row in trace_rows
            ])
            self.assertTrue(all(row["l1_compacted"] for row in trace_rows))
            self.assertEqual(2, metrics["l0_messages_written"])
            self.assertEqual(4, metrics["source_messages_input"])
            self.assertEqual(2, metrics["selected_sessions_compacted"])

            backend_rows = [{
                "id": f"b-{index}",
                "session_id": batch["session_id"],
                "role": message["role"],
                "content": message["content"],
                "timestamp": f"2026-01-01T08:00:0{index}Z",
            } for index, message in enumerate(batch["messages"])]
            with mock.patch.object(adapter, "_post", return_value={
                "code": 0,
                "data": {"messages": backend_rows, "total": len(backend_rows)},
            }):
                history = adapter._load_v3_l0_history("conv")

        self.assertEqual({"selected-1", "selected-2"}, set(history.sessions))
        self.assertEqual({1}, {len(rows) for rows in history.sessions.values()})
        self.assertEqual(
            history.by_source_id["selected-1-u1"],
            history.by_source_id["selected-1-a1"],
        )

    def test_episode_v2_adds_typed_lossless_envelope_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
            )
            payloads = []

            with mock.patch.object(
                adapter,
                "_post",
                side_effect=lambda _endpoint, body: (
                    payloads.append(body) or {"code": 0, "data": {}}
                ),
            ):
                adapter.ingest(Conversation("conv", (_session(
                    "Schedule lunch tomorrow at noon.",
                    "Lunch has been scheduled.",
                    intent="calendar_create",
                    domain="calendar",
                    session_id="event-1",
                ),)))

            content = payloads[0]["messages"][0]["content"]
            trace = json.loads(
                (Path(directory) / "construction.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            metrics = adapter.construction_metrics()

        self.assertTrue(content.startswith(
            "[event-1-u1] [source_role=user] Driver:"
        ))
        self.assertIn(
            "[memory_episode scene=calendar type=event action=add "
            "state=confirmed temporal=future]",
            content,
        )
        self.assertEqual(2, trace["schema_version"])
        self.assertTrue(trace["typed_episode_header"])
        self.assertEqual({"add": 1}, metrics["write_actions"])
        self.assertEqual(1.0, metrics["selected_character_rate"])

    def test_navigation_episode_is_compiled_once_and_reattached_by_source(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_TYPED_COCKPIT_EPISODES="true",
                TDAI_EVAL_L0_WINDOW_BEFORE="2",
                TDAI_EVAL_L0_WINDOW_AFTER="12",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
            )
            session = Session(
                "nav-1",
                "2026-01-01T08:00:00Z",
                (
                    Message(
                        "nav-1-u1", "user", "Find the nearest shopping mall.",
                        speaker="Driver", timestamp="2026-01-01T08:00:00Z",
                    ),
                    Message(
                        "nav-1-a1", "assistant",
                        "Midtown Shopping Center or Ravenswood Shopping Center?",
                        speaker="Car Assistant",
                        timestamp="2026-01-01T08:00:05Z",
                    ),
                    Message(
                        "nav-1-u2", "user", "Let's go to Ravenswood then.",
                        speaker="Driver", timestamp="2026-01-01T08:00:10Z",
                    ),
                    Message(
                        "nav-1-a2", "assistant",
                        "Setting directions to 434 Arastradero Rd.",
                        speaker="Car Assistant",
                        timestamp="2026-01-01T08:00:15Z",
                    ),
                ),
                {"source_domain": "navigation"},
            )
            payloads = []
            with mock.patch.object(
                adapter,
                "_post",
                side_effect=lambda _endpoint, body: (
                    payloads.append(body) or {"code": 0, "data": {}}
                ),
            ):
                adapter.ingest(Conversation("conv", (session,)))

            trace = json.loads(
                (Path(directory) / "construction.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            metrics = adapter.construction_metrics()
            episode = trace["typed_cockpit_episode"]
            backend_rows = [{
                "id": f"b-{index}",
                "session_id": payloads[0]["session_id"],
                "role": message["role"],
                "content": message["content"],
                "timestamp": f"2026-01-01T09:00:0{index}Z",
            } for index, message in enumerate(payloads[0]["messages"])]
            # Retrieval normally runs in a fresh process. A second adapter
            # must reconstruct both source-session boundaries and the typed
            # episode from the durable JSONL trace and migrate it once into
            # the process-external typed episode index.
            episode_index_path = Path(directory) / "typed-episodes.sqlite3"
            reader = self._adapter(
                directory,
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_TYPED_COCKPIT_EPISODES="true",
                TDAI_EVAL_L0_WINDOW_BEFORE="2",
                TDAI_EVAL_L0_WINDOW_AFTER="12",
                TDAI_EVAL_L0_WINDOW_MAX_MESSAGES="8",
                TDAI_EVAL_TYPED_EPISODE_INDEX_PATH=str(episode_index_path),
                TDAI_EVAL_TYPED_EPISODE_INDEX_REQUIRED="true",
            )
            with mock.patch.object(reader, "_post", return_value={
                "code": 0,
                "data": {"messages": backend_rows, "total": len(backend_rows)},
            }):
                expanded = reader._expand_v3_l0_windows(
                    "conv",
                    [{
                        "content": backend_rows[1]["content"],
                        "score": 1.0,
                        "source_ids": ["nav-1-a1"],
                        "metadata": {
                            "level": "L0",
                            "backend_message_id": "b-1",
                        },
                    }],
                    query="shopping center",
                )
            reader_metrics = reader.construction_metrics()
            episode_index_created = episode_index_path.is_file()

        self.assertEqual(
            "Ravenswood Shopping Center", episode["slots"]["destination"]
        )
        self.assertEqual("434 Arastradero Rd", episode["slots"]["address"])
        self.assertEqual(1, metrics["typed_cockpit_episodes"])
        self.assertEqual(
            episode,
            expanded[0]["metadata"]["typed_cockpit_episode"],
        )
        self.assertEqual(1, reader_metrics["typed_episode_index_migrations"])
        self.assertTrue(episode_index_created)

    def test_typed_direct_index_tracks_trace_appends_after_first_load(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_TYPED_EPISODE_HEADERS="false",
                TDAI_EVAL_TYPED_COCKPIT_EPISODES="true",
            )
            first = Session(
                "nav-1", "2026-01-01T08:00:00Z", (
                    Message(
                        "nav-1-u", "user", "Find a hospital.",
                        speaker="Driver",
                    ),
                    Message(
                        "nav-1-a", "assistant",
                        "Navigating to Stanford Hospital.",
                        speaker="Car Assistant",
                    ),
                ), {"source_domain": "navigation"},
            )
            second = Session(
                "nav-2", "2026-01-02T08:00:00Z", (
                    Message(
                        "nav-2-u", "user", "Find an airport.",
                        speaker="Driver",
                    ),
                    Message(
                        "nav-2-a", "assistant",
                        "Navigating to SFO Airport.",
                        speaker="Car Assistant",
                    ),
                ), {"source_domain": "navigation"},
            )
            with mock.patch.object(
                adapter, "_post", return_value={"code": 0, "data": {}}
            ):
                adapter.ingest_session("conv", first)
                adapter._load_construction_trace_scopes()
                adapter.ingest_session("conv", second)

            records = adapter._construction_trace_typed_episode_rows[
                ("conv", "agent")
            ]

        self.assertEqual(
            {"nav-1", "nav-2"},
            {record["session_id"] for record in records},
        )

    def test_deferred_episode_activates_only_after_l0_flush_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "typed-episodes.sqlite3"
            environment = {
                "TDAI_EVAL_L1_WRITE_POLICY": "cockpit_episode_v2",
                "TDAI_EVAL_L1_TYPED_EPISODE_HEADERS": "false",
                "TDAI_EVAL_TYPED_COCKPIT_EPISODES": "true",
                "TDAI_EVAL_L0_FLUSH_CONCURRENCY": "2",
                "TDAI_EVAL_TYPED_EPISODE_INDEX_PATH": str(index_path),
                "TDAI_EVAL_TYPED_EPISODE_INDEX_REQUIRED": "true",
            }
            adapter = self._adapter(directory, **environment)
            session = Session(
                "nav-pending", "2026-08-26T08:00:00+08:00", (
                    Message(
                        "nav-pending-u", "user", "Find an airport.",
                        speaker="Driver",
                        timestamp="2026-08-26T08:00:00+08:00",
                    ),
                    Message(
                        "nav-pending-a", "assistant",
                        "Navigating to Hongqiao Airport.",
                        speaker="Car Assistant",
                        timestamp="2026-08-26T08:00:05+08:00",
                    ),
                ), {"source_domain": "navigation"},
            )
            with mock.patch.object(
                adapter, "_post", return_value={"code": 0, "data": {}}
            ) as post:
                adapter.ingest_session("conv", session)
            post.assert_not_called()

            scope = TypedEpisodeScope(
                conversation_id="conv",
                team_id="team",
                agent_id="agent",
                user_id="user",
                task_id="task",
            )
            index = SQLiteTypedEpisodeIndex(index_path)
            pending = index.get(scope, "nav-pending")
            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertFalse(pending.active)
            self.assertEqual((), index.list_active(scope))

            # A separate reader may parse the growing trace, but a missing
            # source-commit marker must not activate the pending row.
            early_reader = self._adapter(directory, **environment)
            early_reader._load_construction_trace_scopes()
            self.assertEqual((), index.list_active(scope))

            with mock.patch.object(
                adapter, "_post", return_value={"code": 0, "data": {}}
            ) as post:
                adapter.finalize("conv")
            self.assertEqual(1, post.call_count)
            active = index.list_active(scope)
            metrics = adapter.construction_metrics()
            trace_rows = [
                json.loads(line)
                for line in (Path(directory) / "construction.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            # A brand-new index can recover only the committed trace record.
            migrated_path = Path(directory) / "migrated.sqlite3"
            migrated_reader = self._adapter(
                directory,
                **{
                    **environment,
                    "TDAI_EVAL_TYPED_EPISODE_INDEX_PATH": str(migrated_path),
                },
            )
            migrated_reader._load_construction_trace_scopes()
            migrated = SQLiteTypedEpisodeIndex(migrated_path).list_active(scope)

        self.assertEqual(1, len(active))
        self.assertEqual(2, active[0].revision)
        self.assertEqual(1, metrics["typed_episode_index_commits"])
        self.assertEqual(0, metrics["typed_episode_index_pending"])
        self.assertFalse(trace_rows[0]["source_committed"])
        self.assertEqual(
            "typed_episode_source_commit", trace_rows[1]["record_type"]
        )
        self.assertEqual(1, len(migrated))
        self.assertEqual(2, migrated[0].revision)

    def test_failed_deferred_flush_keeps_episode_inactive(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "typed-episodes.sqlite3"
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_WRITE_POLICY="cockpit_episode_v2",
                TDAI_EVAL_L1_TYPED_EPISODE_HEADERS="false",
                TDAI_EVAL_TYPED_COCKPIT_EPISODES="true",
                TDAI_EVAL_L0_FLUSH_CONCURRENCY="2",
                TDAI_EVAL_TYPED_EPISODE_INDEX_PATH=str(index_path),
                TDAI_EVAL_TYPED_EPISODE_INDEX_REQUIRED="true",
            )
            session = Session(
                "nav-failed", "2026-08-26T08:00:00+08:00", (
                    Message(
                        "nav-failed-u", "user", "Find a hospital.",
                        speaker="Driver",
                    ),
                    Message(
                        "nav-failed-a", "assistant",
                        "Navigating to Stanford Hospital.",
                        speaker="Car Assistant",
                    ),
                ), {"source_domain": "navigation"},
            )
            with mock.patch.object(
                adapter, "_post", return_value={"code": 0, "data": {}}
            ):
                adapter.ingest_session("conv", session)
            with mock.patch.object(
                adapter, "_post", side_effect=RuntimeError("L0 unavailable")
            ):
                with self.assertRaises(RuntimeError):
                    adapter.finalize("conv")

            scope = TypedEpisodeScope(
                "conv", "team", "agent", "user", "task"
            )
            index = SQLiteTypedEpisodeIndex(index_path)
            record = index.get(scope, "nav-failed")
            remaining = index.list_active(scope)
            trace_rows = (Path(directory) / "construction.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            metrics = adapter.construction_metrics()

        self.assertIsNotNone(record)
        assert record is not None
        self.assertFalse(record.active)
        self.assertEqual((), remaining)
        self.assertEqual(1, len(trace_rows))
        self.assertEqual(1, metrics["typed_episode_index_pending"])
        self.assertEqual(0, metrics["typed_episode_index_commits"])

    def test_dirty_only_readiness_skips_clean_profile_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            trace_path.write_text(json.dumps({
                "conversation_id": "conv",
                "session_id": "session",
                "agent_id": "agent",
                "pipeline_user_rounds": 0,
                "decision": {"extract_l1": False},
            }) + "\n", encoding="utf-8")
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_L23_READINESS_MODE="dirty_only",
                TDAI_EVAL_READY_SETTLE_SECONDS="300",
                TDAI_EVAL_CLEAN_READY_SETTLE_SECONDS="0",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v2/pipeline/status":
                    return {"data": {
                        "l1": {"queued": 0, "running": 0},
                        "l2": {"queued": 0, "running": 0},
                        "l3": {"queued": 0, "running": 0},
                        "worker": {},
                    }}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post:
                adapter.wait_until_ready("conv", timeout=1)

        self.assertEqual(
            ["/v2/pipeline/status"],
            [call.args[0] for call in post.call_args_list],
        )

    def test_dirty_only_readiness_requires_profiles_for_dirty_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            trace_path.write_text(json.dumps({
                "conversation_id": "conv",
                "session_id": "session",
                "agent_id": "agent",
                "pipeline_user_rounds": 1,
                "decision": {"extract_l1": True},
            }) + "\n", encoding="utf-8")
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_L23_READINESS_MODE="dirty_only",
                TDAI_EVAL_READY_SETTLE_SECONDS="0",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v2/pipeline/status":
                    return {"data": {
                        "l1": {"queued": 0, "running": 0},
                        "l2": {"queued": 0, "running": 0},
                        "l3": {"queued": 0, "running": 0},
                        "worker": {},
                    }}
                if endpoint in {"/v3/scenario/count", "/v3/core/count"}:
                    return {"data": {"total": 1}}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post:
                adapter.wait_until_ready("conv", timeout=1)

        endpoints = [call.args[0] for call in post.call_args_list]
        self.assertIn("/v3/scenario/count", endpoints)
        self.assertIn("/v3/core/count", endpoints)

    def test_scope_readiness_ignores_foreign_pipeline_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            trace_path.write_text(json.dumps({
                "conversation_id": "conv",
                "session_id": "session",
                "agent_id": "agent",
                "pipeline_user_rounds": 1,
                "decision": {"extract_l1": True},
            }) + "\n", encoding="utf-8")
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_L23_READINESS_MODE="dirty_only",
                TDAI_EVAL_READY_SETTLE_SECONDS="0",
            )

            def fake_post(endpoint, _body):
                if endpoint == "/v2/pipeline/status":
                    busy = {
                        "queued": 1,
                        "running": 0,
                        "queued_sessions": ["foreign-session"],
                        "running_sessions": [],
                    }
                    return {"data": {
                        "l1": busy, "l2": busy, "l3": busy, "worker": {},
                    }}
                if endpoint in {"/v3/scenario/count", "/v3/core/count"}:
                    return {"data": {"total": 1}}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                adapter.wait_until_ready("conv", timeout=1)

    def test_scope_readiness_waits_for_owned_pipeline_task(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            trace_path.write_text(json.dumps({
                "conversation_id": "conv",
                "session_id": "session",
                "agent_id": "agent",
                "pipeline_user_rounds": 1,
                "decision": {"extract_l1": True},
            }) + "\n", encoding="utf-8")
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1",
                TDAI_EVAL_READY_SETTLE_SECONDS="0",
            )
            owned = next(iter(adapter._pipeline_scope_sessions("conv") or ()))
            snapshots = iter((
                {"queued": 1, "running": 0,
                 "queued_sessions": [owned], "running_sessions": []},
                {"queued": 0, "running": 0,
                 "queued_sessions": [], "running_sessions": []},
            ))

            def fake_post(endpoint, _body):
                self.assertEqual("/v2/pipeline/status", endpoint)
                return {"data": {"l1": next(snapshots), "worker": {}}}

            with mock.patch.object(adapter, "_post", side_effect=fake_post), \
                    mock.patch("time.sleep"):
                adapter.wait_until_ready("conv", timeout=1)

    def test_invalid_trace_fails_closed_to_required_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "construction.jsonl"
            trace_path.write_text("not-json\n", encoding="utf-8")
            adapter = self._adapter(
                directory,
                TDAI_EVAL_MEMORY_LAYERS="L0,L1,L2,L3",
                TDAI_EVAL_L23_READINESS_MODE="dirty_only",
            )

            def fake_post(endpoint, _body):
                if endpoint in {"/v3/scenario/count", "/v3/core/count"}:
                    return {"data": {"total": 1}}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post) as post:
                self.assertTrue(adapter._profile_layers_ready("conv"))

        self.assertEqual(
            ["/v3/scenario/count", "/v3/core/count"],
            [call.args[0] for call in post.call_args_list],
        )

    @staticmethod
    def _write_dirty_trace(directory: str) -> None:
        trace_path = Path(directory) / "construction.jsonl"
        trace_path.write_text(json.dumps({
            "conversation_id": "conv",
            "session_id": "selected-1",
            "agent_id": "agent",
            "task_id": "task",
            "transport_session_id": "transport-session",
            "source_ids": ["selected-1-u1", "selected-1-a1"],
            "pipeline_user_rounds": 1,
            "decision": {"extract_l1": True},
        }) + "\n", encoding="utf-8")

    @staticmethod
    def _selected_source_query(copies: int = 1) -> dict:
        content = (
            "[selected-1-u1] Driver: Schedule lunch tomorrow at noon.\n"
            "[selected-1-a1] Car Assistant: Lunch has been scheduled."
        )
        messages = [
            {
                "id": f"copy-{index}",
                "session_id": "transport-session",
                "role": "user",
                "content": content,
                "timestamp": f"2026-01-01T08:00:{index:02d}Z",
            }
            for index in range(copies)
        ]
        return {
            "code": 0,
            "data": {"messages": messages, "total": len(messages)},
        }

    def test_zero_output_audit_accepts_existing_atomic_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES="2",
            )
            self._write_dirty_trace(directory)
            conversation = Conversation("conv", (_session(
                "Schedule lunch tomorrow at noon.",
                "Lunch has been scheduled.",
                domain="calendar",
                session_id="selected-1",
            ),))

            with mock.patch.object(adapter, "_post", return_value={
                "code": 0, "data": {"total": 1},
            }) as post:
                result = adapter.ensure_construction(conversation, timeout=1)

        self.assertEqual("verified", result["status"])
        self.assertEqual(0, result["repair_attempts"])
        self.assertEqual(["/v3/atomic/count"], [
            call.args[0] for call in post.call_args_list
        ])

    def test_zero_output_audit_replays_selected_batch_then_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES="2",
                TDAI_EVAL_L1_READY_SETTLE_SECONDS="0",
            )
            self._write_dirty_trace(directory)
            conversation = Conversation("conv", (_session(
                "Schedule lunch tomorrow at noon.",
                "Lunch has been scheduled.",
                domain="calendar",
                session_id="selected-1",
            ),))
            counts = iter((0, 0, 1))
            replay_payloads = []

            def fake_post(endpoint, body):
                if endpoint == "/v3/atomic/count":
                    return {"code": 0, "data": {"total": next(counts)}}
                if endpoint == "/v2/pipeline/status":
                    return {"code": 0, "data": {
                        "l1": {
                            "queued": 0, "running": 0,
                            "queued_sessions": [], "running_sessions": [],
                        },
                        "worker": {},
                    }}
                if endpoint == "/v3/conversation/query":
                    return self._selected_source_query()
                if endpoint == "/v3/conversation/add":
                    replay_payloads.append(body)
                    return {"code": 0, "data": {}}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                result = adapter.ensure_construction(conversation, timeout=1)

        self.assertEqual("repaired", result["status"])
        self.assertEqual(1, result["repair_attempts"])
        self.assertEqual(1, result["replayed_messages"])
        self.assertEqual(1, len(replay_payloads))
        self.assertEqual("user", replay_payloads[0]["messages"][0]["role"])
        self.assertIn(
            "Schedule lunch tomorrow",
            replay_payloads[0]["messages"][0]["content"],
        )

    def test_zero_output_audit_fails_after_bounded_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES="1",
                TDAI_EVAL_L1_READY_SETTLE_SECONDS="0",
            )
            self._write_dirty_trace(directory)
            conversation = Conversation("conv", (_session(
                "Schedule lunch tomorrow at noon.",
                "Lunch has been scheduled.",
                domain="calendar",
                session_id="selected-1",
            ),))

            def fake_post(endpoint, _body):
                if endpoint == "/v3/atomic/count":
                    return {"code": 0, "data": {"total": 0}}
                if endpoint == "/v2/pipeline/status":
                    return {"code": 0, "data": {
                        "l1": {
                            "queued": 0, "running": 0,
                            "queued_sessions": [], "running_sessions": [],
                        },
                        "worker": {},
                    }}
                if endpoint == "/v3/conversation/query":
                    return self._selected_source_query()
                if endpoint == "/v3/conversation/add":
                    return {"code": 0, "data": {}}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                with self.assertRaisesRegex(RuntimeError, "remained empty"):
                    adapter.ensure_construction(conversation, timeout=1)

    def test_zero_output_l0_only_reuses_observed_replay_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES="1",
                TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL="l0_only",
                TDAI_EVAL_L1_READY_SETTLE_SECONDS="0",
            )
            self._write_dirty_trace(directory)
            conversation = Conversation("conv", (_session(
                "Schedule lunch tomorrow at noon.",
                "Lunch has been scheduled.",
                domain="calendar",
                session_id="selected-1",
            ),))

            def fake_post(endpoint, _body):
                if endpoint == "/v3/atomic/count":
                    return {"code": 0, "data": {"total": 0}}
                if endpoint == "/v2/pipeline/status":
                    return {"code": 0, "data": {
                        "l1": {
                            "queued": 0, "running": 0,
                            "queued_sessions": [], "running_sessions": [],
                        },
                        "worker": {},
                    }}
                if endpoint == "/v3/conversation/query":
                    return self._selected_source_query(copies=3)
                if endpoint == "/v3/conversation/add":
                    raise AssertionError("observed replay must not run again")
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                result = adapter.ensure_construction(conversation, timeout=1)

        self.assertEqual("l0_only_after_empty_l1", result["status"])
        self.assertEqual(2, result["repair_attempts"])
        self.assertEqual(0, result["repair_attempts_this_process"])
        perspective = adapter._perspective_isolations("conv")[0]
        self.assertFalse(adapter._profile_scope_dirty("conv", perspective))

    def test_zero_output_async_replay_returns_without_second_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self._adapter(
                directory,
                TDAI_EVAL_L1_BATCH_MODE="conversation",
                TDAI_EVAL_L1_COMPACT_SELECTED_SESSIONS="true",
                TDAI_EVAL_L1_ZERO_OUTPUT_RETRIES="1",
                TDAI_EVAL_L1_ZERO_OUTPUT_TERMINAL="l0_only",
                TDAI_EVAL_L1_ZERO_OUTPUT_WAIT_FOR_REPAIR="false",
                TDAI_EVAL_L1_READY_SETTLE_SECONDS="0",
            )
            self._write_dirty_trace(directory)
            conversation = Conversation("conv", (_session(
                "Schedule lunch tomorrow at noon.",
                "Lunch has been scheduled.",
                domain="calendar",
                session_id="selected-1",
            ),))
            counts = iter((0, 0))
            status_calls = 0
            replay_payloads = []

            def fake_post(endpoint, body):
                nonlocal status_calls
                if endpoint == "/v3/atomic/count":
                    return {"code": 0, "data": {"total": next(counts)}}
                if endpoint == "/v2/pipeline/status":
                    status_calls += 1
                    return {"code": 0, "data": {
                        "l1": {
                            "queued": 0, "running": 0,
                            "queued_sessions": [], "running_sessions": [],
                        },
                        "worker": {},
                    }}
                if endpoint == "/v3/conversation/query":
                    return self._selected_source_query()
                if endpoint == "/v3/conversation/add":
                    replay_payloads.append(body)
                    return {"code": 0, "data": {}}
                raise AssertionError(endpoint)

            with mock.patch.object(adapter, "_post", side_effect=fake_post):
                result = adapter.ensure_construction(conversation, timeout=1)

        self.assertEqual("l0_only_repair_enqueued", result["status"])
        self.assertEqual(1, result["repair_attempts"])
        self.assertEqual(1, result["repair_attempts_this_process"])
        self.assertEqual(1, status_calls)
        self.assertEqual(1, len(replay_payloads))

    def test_rejects_unknown_l23_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "TDAI_EVAL_L23_SCHEDULE"):
                self._adapter(
                    directory, TDAI_EVAL_L23_SCHEDULE="unknown-schedule"
                )


if __name__ == "__main__":
    unittest.main()
