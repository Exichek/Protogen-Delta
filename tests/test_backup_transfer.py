"""SSH-копия не считается облачной и проверяется до записи квитанции."""

import ast
import importlib.util
import json
import shutil
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from test_backup_snapshot import source_data

from protogen_delta.backup_snapshot import create_online_snapshot, pack_snapshot

spec = importlib.util.spec_from_file_location(
    "backup_transfer_test", Path(__file__).parents[1] / "scripts/backup_delta.py"
)
assert spec and spec.loader
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


@pytest.mark.parametrize("broken", [False, True])
def test_transfer_checks_bytes_and_actual_restoration_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bool
) -> None:
    counts = create_online_snapshot(source_data(tmp_path), tmp_path / "snapshot")
    original = tmp_path / "server.zip"
    digest = pack_snapshot(tmp_path / "snapshot", original)
    seen = []

    def run(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        seen.append(args)
        if args[0] == "ssh":
            payload = kwargs["input"]
            compile(payload, "remote-backup", "exec")
            assignments = {
                n.targets[0].id: ast.literal_eval(n.value)
                for n in ast.parse(payload).body[:3]
            }
            compile(assignments["CHILD"], "container-backup", "exec")
            info = counts | {
                "sha256": digest,
                "remote_archive": "/srv/delta/.secrets/backup-transfers/"
                + assignments["JOB"]
                + ".zip",
            }
            return CompletedProcess(args, 0, json.dumps(info), "")
        assert args[0] == "scp"
        shutil.copyfile(original, args[-1])
        if broken:
            Path(args[-1]).write_bytes(b"broken")
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(transfer.subprocess, "run", run)
    if broken:
        with pytest.raises(ValueError, match="SHA-256"):
            transfer.capture_backup(
                "owner@example.test", "/srv/delta", tmp_path / "copies"
            )
        assert not list((tmp_path / "copies").glob("*.receipt.json"))
    else:
        receipt = transfer.capture_backup(
            "owner@example.test", "/srv/delta", tmp_path / "copies"
        )
        assert receipt["sha256"] == digest and receipt["restoration_verified"]
        assert receipt["cloud_upload_verified"] is False
        assert receipt["databases"] == 2
        assert len(list((tmp_path / "copies").glob("*.receipt.json"))) == 1
    assert len(seen) == 2


@pytest.mark.parametrize(
    "host,root",
    [
        ("-oProxyCommand=bad", "/srv/delta"),
        ("owner@-bad", "/srv/delta"),
        ("good", "/srv/../etc"),
        ("good", "relative/path"),
    ],
)
def test_transfer_rejects_shell_options_and_unexpected_paths(
    tmp_path: Path, host: str, root: str
) -> None:
    with pytest.raises(ValueError):
        transfer.capture_backup(host, root, tmp_path / "copies")
    assert not (tmp_path / "copies").exists()
