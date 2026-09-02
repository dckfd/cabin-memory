from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class LabeledInterval:
    session: str
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_textgrid(path: Path) -> list[LabeledInterval]:
    session = path.parent.name
    speaker = ""
    start: float | None = None
    end: float | None = None
    intervals: list[LabeledInterval] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("name ="):
            speaker = str(ast.literal_eval(line.split("=", 1)[1].strip()))
        elif line.startswith("intervals ["):
            start = None
            end = None
        elif line.startswith("xmin =") and start is None:
            start = float(line.split("=", 1)[1])
        elif line.startswith("xmax =") and start is not None and end is None:
            end = float(line.split("=", 1)[1])
        elif line.startswith("text =") and start is not None and end is not None:
            text = str(ast.literal_eval(line.split("=", 1)[1].strip())).strip()
            if speaker and text:
                intervals.append(LabeledInterval(session, speaker, start, end, text))
            start = None
            end = None
    return intervals


def _scoreable_length(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))


def _overlap(left: LabeledInterval, right: LabeledInterval) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def select_intervals(
    dev_root: Path, *, count: int, minimum_seconds: float = 1.0,
    maximum_seconds: float = 8.0,
) -> list[LabeledInterval]:
    candidates: list[LabeledInterval] = []
    for label in sorted(dev_root.glob("*/DX01C01.TextGrid")):
        session_rows = parse_textgrid(label)
        for row in session_rows:
            if not minimum_seconds <= row.duration <= maximum_seconds:
                continue
            if not 5 <= _scoreable_length(row.text) <= 48:
                continue
            if any(
                other.speaker != row.speaker and _overlap(row, other) > 0.08
                for other in session_rows
            ):
                continue
            candidates.append(row)
    candidates.sort(key=lambda row: hashlib.sha256(
        f"{row.session}|{row.speaker}|{row.start:.3f}|{row.text}".encode("utf-8")
    ).hexdigest())
    selected: list[LabeledInterval] = []
    per_session: dict[str, int] = {}
    # First pass prevents one long session from dominating the slice.
    for limit in range(1, 20):
        for row in candidates:
            if row in selected or per_session.get(row.session, 0) >= limit:
                continue
            selected.append(row)
            per_session[row.session] = per_session.get(row.session, 0) + 1
            if len(selected) == count:
                return selected
    raise ValueError(f"AISHELL-5 yielded {len(selected)} clips; requested {count}")


def extract_clips(
    dev_root: Path, intervals: Iterable[LabeledInterval], output_dir: Path,
    *, channel_mode: str = "rotate",
) -> list[dict[str, Any]]:
    if channel_mode not in {"rotate", "all"}:
        raise ValueError(f"unsupported channel mode: {channel_mode}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, interval in enumerate(intervals):
        channels = range(1, 5) if channel_mode == "all" else (index % 4 + 1,)
        source_utterance_id = f"aishell5-dev-{interval.session}-{index:04d}"
        for channel in channels:
            source = dev_root / interval.session / f"DX0{channel}C01.wav"
            if not source.is_file():
                raise FileNotFoundError(source)
            clip_id = f"{source_utterance_id}-ch{channel}"
            target = output_dir / f"{clip_id}.wav"
            pad_start = max(0.0, interval.start - 0.12)
            pad_end = interval.end + 0.12
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{pad_start:.3f}", "-t", f"{pad_end - pad_start:.3f}",
                "-i", str(source), "-ac", "1", "-ar", "16000", str(target),
            ]
            subprocess.run(command, check=True)
            timestamp = origin + timedelta(days=int(interval.session), seconds=interval.start)
            ended = timestamp + timedelta(seconds=interval.duration)
            rows.append({
                "clip_id": clip_id,
                "source_utterance_id": source_utterance_id,
                "audio_path": str(target.resolve()),
                "reference": interval.text,
                "speaker_id": interval.speaker,
                # Required synthetic fixture value: AISHELL-5 identifies speakers,
                # but does not establish their seat or cockpit role.
                "speaker_role": "driver",
                "speaker_role_note": "synthetic fixture; not an AISHELL-5 source label",
                "namespace": "aishell5-public-shadow",
                "conversation_id": f"aishell5-dev-{interval.session}",
                "started_at": timestamp.isoformat().replace("+00:00", "Z"),
                "ended_at": ended.isoformat().replace("+00:00", "Z"),
                "vehicle_id": "aishell5-public",
                "seat": "unknown",
                "source_session": interval.session,
                "source_channel": channel,
                "source_interval": [interval.start, interval.end],
                "timestamp_note": "synthetic ordering; not source recording time",
            })
    return rows


def _normalized_distance(left: str, right: str) -> float:
    from .cockpit_public_voice_eval import edit_distance, normalize_asr_text

    normalized_left = normalize_asr_text(left)
    normalized_right = normalize_asr_text(right)
    denominator = max(1, len(normalized_left), len(normalized_right))
    return edit_distance(normalized_left, normalized_right) / denominator


def fuse_multichannel_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a reference-free transcript medoid for every four-channel utterance."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        utterance_id = str(row.get("source_utterance_id") or "")
        if not utterance_id:
            raise ValueError("multichannel row is missing source_utterance_id")
        grouped.setdefault(utterance_id, []).append(row)

    fused: list[dict[str, Any]] = []
    for utterance_id, candidates in grouped.items():
        if len(candidates) != 4 or {row.get("source_channel") for row in candidates} != {1, 2, 3, 4}:
            raise ValueError(f"{utterance_id} does not contain exactly channels 1-4")

        def rank(candidate: dict[str, Any]) -> tuple[float, int, int]:
            hypothesis = str(candidate.get("hypothesis") or "")
            consensus_cost = sum(
                _normalized_distance(hypothesis, str(other.get("hypothesis") or ""))
                for other in candidates
            )
            # Prefer more informative text, then a stable channel order, on ties.
            return consensus_cost, -len(hypothesis.replace(" ", "")), int(candidate["source_channel"])

        selected = min(candidates, key=rank)
        output = dict(selected)
        output["clip_id"] = utterance_id
        output["fusion_method"] = "four-channel-transcript-medoid-v1"
        output["selected_source_channel"] = selected["source_channel"]
        output["channel_hypotheses"] = {
            str(row["source_channel"]): str(row.get("hypothesis") or "")
            for row in sorted(candidates, key=lambda item: int(item["source_channel"]))
        }
        fused.append(output)
    return fused


def transcribe_funasr(rows: list[dict[str, Any]], model_name: str) -> None:
    from funasr import AutoModel

    model = AutoModel(
        model=model_name,
        device="cuda:0",
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )
    for row in rows:
        started = time.perf_counter()
        result = model.generate(
            input=row["audio_path"],
            batch_size_s=60,
            hotword="",
        )
        row["hypothesis"] = str((result[0] if result else {}).get("text") or "").strip()
        row["asr_seconds"] = time.perf_counter() - started
        row["source_system"] = f"funasr:{model_name}"
        row["transcript_confidence"] = None


def transcribe_faster_whisper(rows: list[dict[str, Any]], model_name: str) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    for row in rows:
        started = time.perf_counter()
        segments, info = model.transcribe(
            row["audio_path"], language="zh", beam_size=5,
            condition_on_previous_text=False, vad_filter=False,
        )
        row["hypothesis"] = "".join(segment.text for segment in segments).strip()
        row["asr_seconds"] = time.perf_counter() - started
        row["source_system"] = f"faster-whisper:{model_name}"
        row["transcript_confidence"] = float(getattr(info, "language_probability", 0.0))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--backend", choices=("funasr", "faster-whisper"), default="funasr")
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--channel-mode", choices=("rotate", "all"), default="rotate")
    parser.add_argument("--fused-output", type=Path)
    args = parser.parse_args()

    if args.fused_output and args.channel_mode != "all":
        parser.error("--fused-output requires --channel-mode all")

    intervals = select_intervals(args.dev_root, count=args.count)
    rows = extract_clips(
        args.dev_root, intervals, args.clips_dir, channel_mode=args.channel_mode,
    )
    if args.backend == "funasr":
        transcribe_funasr(rows, args.model)
    else:
        transcribe_faster_whisper(rows, args.model)
    _write_jsonl(args.output, rows)
    if args.fused_output:
        _write_jsonl(args.fused_output, fuse_multichannel_rows(rows))
    compact = {
        "clips": len(rows),
        "utterances": len(intervals),
        "backend": args.backend,
        "model": args.model,
        "channel_mode": args.channel_mode,
        "sessions": len({row["source_session"] for row in rows}),
        "channels": sorted({row["source_channel"] for row in rows}),
        "audio_seconds": sum(
            float(row["source_interval"][1]) - float(row["source_interval"][0])
            for row in rows
        ),
        "asr_wall_seconds": sum(float(row["asr_seconds"]) for row in rows),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
