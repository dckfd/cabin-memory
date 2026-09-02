from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


_INTERNAL_ARTIFACTS = re.compile(
    r"(?:\bevidence\s*#?\s*\d*|\bsource[_ -]?id\b|\bcitation\b|"
    r"tdai_(?:evidence|recall|memory|conversation)|<\/?tdai_[^>]*>|证据\s*(?:编号|#?\d+))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerQualityResult:
    passed: bool
    violations: tuple[str, ...]


def audit_user_facing_answer(
    answer: str,
    *,
    required_term_groups: Sequence[Sequence[str]] = (),
    known_subjects_with_other_history: Sequence[str] = (),
    forbidden_attributions: Sequence[Mapping[str, Any]] = (),
) -> AnswerQualityResult:
    """Strict deterministic audit; contracts are evaluation data, not prompts."""
    violations: list[str] = []
    if _INTERNAL_ARTIFACTS.search(answer):
        violations.append("internal_evidence_artifact")

    folded = answer.casefold()
    for index, alternatives in enumerate(required_term_groups):
        if not any(str(term).casefold() in folded for term in alternatives):
            violations.append(f"required_field_missing:{index}")

    for subject in known_subjects_with_other_history:
        escaped = re.escape(subject)
        broad_patterns = (
            rf"(?:没有|无).{{0,12}}(?:关于|有关)?\s*{escaped}.{{0,12}}(?:信息|记录|历史)",
            rf"(?:没有|无).{{0,12}}(?:名为|叫)\s*{escaped}.{{0,12}}(?:的人|个人|信息)",
            rf"no information about(?: an individual named)?\s+{escaped}\b",
            rf"no (?:history|records?) (?:for|about)\s+{escaped}\b",
        )
        if any(re.search(pattern, answer, re.IGNORECASE) for pattern in broad_patterns):
            violations.append(f"abstention_scope_too_broad:{subject}")

    for rule in forbidden_attributions:
        subject = str(rule.get("subject") or "").strip()
        attributes = [str(item) for item in rule.get("attributes", []) if str(item)]
        if not subject or not attributes:
            continue
        subject_pos = folded.find(subject.casefold())
        if subject_pos < 0:
            continue
        for attribute in attributes:
            attribute_pos = folded.find(attribute.casefold(), subject_pos)
            if 0 <= attribute_pos - subject_pos <= int(rule.get("window", 60)):
                violations.append(f"forbidden_attribution:{subject}:{attribute}")

    return AnswerQualityResult(not violations, tuple(dict.fromkeys(violations)))


def audit_jsonl(
    rows: Iterable[Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    results = []
    for row in rows:
        qa_id = str(row.get("qa_id") or "")
        answer = str(row.get("predicted_answer") or row.get("answer") or "")
        contract = contracts.get(qa_id, {})
        audit = audit_user_facing_answer(
            answer,
            required_term_groups=contract.get("required_term_groups", ()),
            known_subjects_with_other_history=contract.get(
                "known_subjects_with_other_history", ()
            ),
            forbidden_attributions=contract.get("forbidden_attributions", ()),
        )
        results.append({
            "qa_id": qa_id,
            "answer": answer,
            "passed": audit.passed,
            "violations": list(audit.violations),
        })
    return {
        "schema_version": 1,
        "protocol": "cockpit-user-facing-answer-quality-v1",
        "count": len(results),
        "passed": sum(row["passed"] for row in results),
        "failed_ids": [row["qa_id"] for row in results if not row["passed"]],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.answers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contracts = json.loads(args.contracts.read_text(encoding="utf-8"))["contracts"]
    summary = audit_jsonl(rows, contracts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False))
    return 0 if summary["passed"] == summary["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
