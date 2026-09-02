from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .plugins import PluginCatalog


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "benchmarks/framework_eval/profiles.json"
DATASETS = ROOT / "benchmarks/framework_eval/datasets.json"


class TencentDBProvisioner:
    def __init__(self, base_url: str, api_key: str, service_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.service_id = service_id

    def post(self, endpoint: str, body: dict, *, user_key: str = "") -> dict:
        headers = {
            "Content-Type": "application/json",
            "x-tdai-service-id": self.service_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if user_key:
            headers["x-tdai-user-key"] = user_key
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"TencentDB provisioning request {endpoint} failed: "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        if result.get("code") not in (None, 0):
            raise RuntimeError(
                f"TencentDB provisioning request {endpoint} failed: {result}"
            )
        return dict(result.get("data") or {})


def load_reused_principal(
    manifest_path: Path, key_path: Path, *, service_id: str,
) -> tuple[str, str, str]:
    """Return team, user and key while keeping the key out of run artifacts."""
    principal_path = manifest_path.resolve()
    principal = json.loads(principal_path.read_text(encoding="utf-8"))
    team_id = str(principal.get("team_id") or "")
    user_id = str(principal.get("user_id") or "")
    principal_service = str(principal.get("service_id") or "default")
    if not team_id or not user_id:
        raise ValueError(
            f"principal manifest has no team_id/user_id: {principal_path}"
        )
    if principal_service != service_id:
        raise ValueError(
            "reused principal service_id does not match --service-id: "
            f"{principal_service!r} != {service_id!r}"
        )
    user_key = key_path.resolve().read_text(encoding="utf-8").strip()
    if not user_key:
        raise ValueError(f"existing user key file is empty: {key_path.resolve()}")
    return team_id, user_id, user_key


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision fresh isolated TencentDB namespaces"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="TDAI_API_KEY")
    parser.add_argument(
        "--allow-empty-api-key", action="store_true",
        help="Allow a local MemoryCore deployment whose Bearer gate is disabled",
    )
    parser.add_argument(
        "--admin-user-key-file", type=Path,
        help=(
            "Use an existing administrator user key instead of calling "
            "init-admin; the key is read but never written to run artifacts"
        ),
    )
    parser.add_argument(
        "--user-key-file", type=Path,
        help=(
            "Persist the newly created benchmark user's key in a separate "
            "0600 file; only the path, never the key, is reported"
        ),
    )
    parser.add_argument(
        "--reuse-principal-manifest", type=Path,
        help=(
            "Reuse team/user metadata from an existing isolation manifest "
            "while creating fresh agent/task namespaces"
        ),
    )
    parser.add_argument(
        "--existing-user-key-file", type=Path,
        help="User-key input paired with --reuse-principal-manifest",
    )
    parser.add_argument("--service-id", default="default")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conversation", action="append", default=[])
    parser.add_argument(
        "--selection-manifest", type=Path,
        help="Protocol manifest containing conversation_ids",
    )
    parser.add_argument(
        "--all-conversations", action="store_true",
        help="Provision every conversation exposed by the selected dataset",
    )
    parser.add_argument("--dataset", default="locomo_refined")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--memory-layers", default="L0,L1,L2,L3",
        help="Comma-separated layers recorded in the isolation manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit(
            f"refusing to overwrite existing isolation manifest: {args.output}"
        )
    api_key = os.getenv(args.api_key_env, "")
    if not api_key and not args.allow_empty_api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    memory_layers = list(dict.fromkeys(
        item.strip().upper() for item in args.memory_layers.split(",") if item.strip()
    ))
    if not memory_layers or set(memory_layers) - {"L0", "L1", "L2", "L3"}:
        raise SystemExit("--memory-layers must contain only L0,L1,L2,L3")
    layer_label = "/".join(memory_layers)

    catalog = PluginCatalog(
        root=ROOT,
        framework_profiles=PROFILES,
        dataset_profiles=DATASETS,
    )
    dataset = catalog.create_dataset(args.dataset, root=args.dataset_root)
    selectors = sum(bool(value) for value in (
        args.all_conversations, args.conversation, args.selection_manifest,
    ))
    if selectors > 1:
        raise SystemExit(
            "choose exactly one of --conversation, --selection-manifest, or "
            "--all-conversations"
        )
    if args.selection_manifest:
        selection = json.loads(
            args.selection_manifest.read_text(encoding="utf-8")
        )
        manifest_dataset = str(selection.get("dataset_id") or "")
        if manifest_dataset and manifest_dataset != args.dataset:
            raise SystemExit(
                "selection manifest dataset does not match --dataset: "
                f"{manifest_dataset!r} != {args.dataset!r}"
            )
        requested_conversations = list(dict.fromkeys(
            str(value) for value in selection.get("conversation_ids") or []
        ))
    else:
        requested_conversations = (
            [item.conversation_id for item in dataset.conversations()]
            if args.all_conversations else list(args.conversation)
        )
    if not requested_conversations:
        raise SystemExit(
            "provide --conversation, --selection-manifest, or "
            "--all-conversations"
        )
    # Validate the entire requested dataset slice before the first remote
    # mutation.  A typo must not leave an otherwise fresh benchmark volume
    # partially provisioned with orphaned users or teams.
    conversation_speakers: dict[str, list[str]] = {}
    for conversation_id in requested_conversations:
        try:
            conversation = dataset.conversation(conversation_id)
        except KeyError as exc:
            known = ", ".join(
                item.conversation_id for item in dataset.conversations()
            )
            raise SystemExit(
                f"unknown conversation {conversation_id!r}; available: {known}"
            ) from exc
        speakers = list(dict.fromkeys(
            str(conversation.metadata.get(key) or "").strip()
            for key in ("speaker_a", "speaker_b")
            if str(conversation.metadata.get(key) or "").strip()
        ))
        conversation_speakers[conversation_id] = speakers

    if bool(args.reuse_principal_manifest) != bool(args.existing_user_key_file):
        raise SystemExit(
            "--reuse-principal-manifest and --existing-user-key-file must be "
            "provided together"
        )
    if args.reuse_principal_manifest and (
        args.admin_user_key_file or args.user_key_file
    ):
        raise SystemExit(
            "principal reuse cannot be combined with --admin-user-key-file or "
            "--user-key-file"
        )

    client = TencentDBProvisioner(args.base_url, api_key, args.service_id)
    # A fresh evaluation volume has an empty v3 metadata database.  Bootstrap
    # its system administrator once, then use that administrator only to
    # create a normal benchmark principal.  User keys are intentionally kept
    # in process memory and never written to the isolation manifest.
    principal_reused_from: str | None = None
    if args.reuse_principal_manifest:
        principal_path = args.reuse_principal_manifest.resolve()
        key_path = args.existing_user_key_file.resolve()
        try:
            team_id, user_id, user_key = load_reused_principal(
                principal_path, key_path, service_id=args.service_id
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        principal_reused_from = str(principal_path)
    else:
        if args.admin_user_key_file:
            admin_key = args.admin_user_key_file.read_text(
                encoding="utf-8"
            ).strip()
            if not admin_key:
                raise SystemExit(
                    "administrator user key file is empty: "
                    f"{args.admin_user_key_file}"
                )
        else:
            admin = client.post("/v3/internal/meta/user/init-admin", {
                "username": f"framework-eval-admin-{args.run_id}",
            })
            admin_key = str(admin["user_key"])
        user = client.post("/v3/meta/user/create", {
            "username": f"framework-eval-{args.run_id}",
        }, user_key=admin_key)
        user_id = str(user["user_id"])
        user_key = str(user["default_user_key"])
        if args.user_key_file:
            user_key_path = args.user_key_file.resolve()
            user_key_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(
                    user_key_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError as exc:
                raise SystemExit(
                    f"refusing to overwrite user key file: {user_key_path}"
                ) from exc
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(user_key + "\n")
        team = client.post("/v3/meta/team/create", {
            "name": f"framework-eval-{args.run_id}",
            "description": "Fresh full-layer memory evaluation namespace",
            "owner_user_id": user_id,
        }, user_key=user_key)
        team_id = str(team["team_id"])

    conversations: dict[str, dict] = {}
    for conversation_id, speakers in conversation_speakers.items():
        perspectives: dict[str, dict] = {}
        # Persona-to-persona benchmarks declare two human memory subjects and
        # retain independent perspectives. Driver-to-assistant datasets are a
        # single user/agent interaction, so both roles belong in one namespace.
        scope_labels: list[str | None] = list(speakers) if len(speakers) >= 2 else [None]
        first_scope: dict | None = None
        for speaker in scope_labels:
            scope_name = speaker or "dialogue"
            agent = client.post("/v3/meta/agent/create", {
                "team_id": team_id,
                "name": f"{args.run_id}-{conversation_id}-{scope_name}",
                "description": (
                    f"Independent {layer_label} memory namespace for "
                    f"{scope_name} in {conversation_id}"
                ),
                "owner_user_id": user_id,
                "visibility": "restricted",
            }, user_key=user_key)
            agent_id = str(agent["agent_id"])
            task = client.post("/v3/meta/task/create", {
                "team_id": team_id,
                "creator_user_id": user_id,
                "title": f"{args.run_id}-{conversation_id}-{scope_name}",
                "description": (
                    f"Independent {layer_label} task namespace paired with the "
                    f"agent namespace for {scope_name}"
                ),
                "source_type": "other",
                "linked_agents": [{
                    "agent_id": agent_id,
                    "role_in_task": "memory_subject",
                }],
            }, user_key=user_key)
            scope = {
                "team_id": team_id,
                "agent_id": agent_id,
                "user_id": user_id,
                "task_id": str(task["task_id"]),
            }
            first_scope = first_scope or scope
            if speaker is not None:
                perspectives[speaker] = scope
        assert first_scope is not None
        conversation_scope = {
            "agent_id": first_scope["agent_id"],
            "task_id": first_scope["task_id"],
        }
        if perspectives:
            conversation_scope["perspectives"] = perspectives
        conversations[conversation_id] = conversation_scope

    manifest = {
        "schema_version": 2,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "team_id": team_id,
        "user_id": user_id,
        "user_key_file": (
            str(args.existing_user_key_file.resolve())
            if args.existing_user_key_file else
            (str(args.user_key_file.resolve()) if args.user_key_file else None)
        ),
        "principal_reused_from": principal_reused_from,
        "service_id": args.service_id,
        "metadata_api_version": "v3",
        "memory_layers": memory_layers,
        "conversations": conversations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "run_id": args.run_id,
        "team_id": team_id,
        "user_id": user_id,
        "conversations": {
            conversation_id: list(row.get("perspectives") or {"single": {}})
            for conversation_id, row in conversations.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
