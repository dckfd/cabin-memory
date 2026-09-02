from __future__ import annotations

from pathlib import Path

from .base import DatasetInfo
from .kvret_memory import KVRETMemoryDataset


class SLURPMemoryDataset(KVRETMemoryDataset):
    """Derived same-speaker memory task over official SLURP transcripts.

    The normalized JSONL schema is intentionally identical to KVRET-Memory's
    canonical boundary, so the tested framework sees the same Conversation and
    Question contracts rather than dataset-specific shortcuts.
    """

    dataset_id = "slurp_memory"
    info = DatasetInfo(
        dataset_id="slurp_memory",
        version="slurp-memory-138-v1",
        modalities=frozenset({"text"}),
        task_types=frozenset({
            "qa", "spoken-language-understanding", "fragmented-command",
            "cross-session", "same-speaker",
        }),
        metadata={"official_slurp_metric": False, "audio_downloaded": False},
    )

    def __init__(self, root: Path) -> None:
        super().__init__(root)

    def validate(self) -> list[str]:
        errors = super().validate()
        source_slurp_ids: set[int] = set()
        for conversation in self.conversations():
            speaker_hashes = {
                str(session.metadata.get("source_speaker_hash") or "")
                for session in conversation.sessions
            }
            if len(speaker_hashes) != 1 or "" in speaker_hashes:
                errors.append(
                    f"conversation {conversation.conversation_id} mixes source speakers"
                )
            for session in conversation.sessions:
                if [message.role for message in session.messages] != [
                    "user", "assistant", "user"
                ]:
                    errors.append(
                        f"session {session.session_id} is not user/assistant/user"
                    )
                source_id = int(session.metadata.get("source_slurp_id") or 0)
                if not source_id or source_id in source_slurp_ids:
                    errors.append(
                        f"session {session.session_id} reuses or omits source_slurp_id"
                    )
                source_slurp_ids.add(source_id)
        return errors
