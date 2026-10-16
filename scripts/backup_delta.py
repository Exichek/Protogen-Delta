"""Получить проверенную копию Docker-бота по SSH и проверить восстановление локально."""

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from protogen_delta.backup_snapshot import file_hash, restore_archive


def capture_backup(host: str, remote_root: str, output: Path) -> dict[str, object]:
    if (
        not re.fullmatch(r"(?:[\w.-]+@)?[\w.-]+", host)
        or host.startswith("-")
        or host.split("@")[-1].startswith("-")
    ):
        raise ValueError("Нужен SSH-хост без параметров командной строки")
    if not re.fullmatch(r"/[\w/.-]+", remote_root) or ".." in remote_root.split("/"):
        raise ValueError(
            "Нужен абсолютный путь репозитория без пробелов и переходов .."
        )
    job = (
        "delta-data-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex[:8]
    )
    output = output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive = output / (job + ".zip")
    child = """import json
from protogen_delta.config.settings import load_settings
from protogen_delta.backup_snapshot import create_online_snapshot,pack_snapshot
data=load_settings().data_dir
folder=data/'backups'/JOB
result=create_online_snapshot(data,folder)
archive=folder.with_suffix('.zip')
result.update({'sha256':pack_snapshot(folder,archive),'container_archive':str(archive)})
print(json.dumps(result))
"""
    remote = """import json,subprocess
from pathlib import Path
root=Path.cwd()
assert str(root)==ROOT
folder=root/'.secrets'/'backup-transfers'
folder.mkdir(mode=0o700,parents=True,exist_ok=True)
target=folder/(JOB+'.zip')
assert not target.exists()
result=subprocess.run(['docker','compose','exec','-T','delta','python','-'],input='JOB='+repr(JOB)+'\\n'+CHILD,text=True,capture_output=True,check=True)
info=json.loads(result.stdout)
assert info['container_archive']=='/app/data/backups/'+JOB+'.zip'
subprocess.run(['docker','cp','protogen-delta-delta-1:'+info['container_archive'],str(target)],check=True,capture_output=True)
target.chmod(0o600)
info['remote_archive']=str(target)
print(json.dumps(info))
"""
    payload = (
        "JOB="
        + repr(job)
        + "\nROOT="
        + repr(remote_root)
        + "\nCHILD="
        + repr(child)
        + "\n"
        + remote
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            host,
            "cd " + shlex.quote(remote_root) + " && python3 -",
        ],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    metadata = json.loads(result.stdout)
    expected = remote_root + "/.secrets/backup-transfers/" + job + ".zip"
    if metadata["remote_archive"] != expected:
        raise ValueError("Неожиданный путь архива")
    subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            host + ":" + expected,
            str(archive),
        ],
        capture_output=True,
        check=True,
        timeout=180,
    )
    archive.chmod(0o600)
    if file_hash(archive) != metadata["sha256"]:
        raise ValueError("SHA-256 загруженной копии не совпадает")
    with TemporaryDirectory(prefix="delta-restore-check-", dir=output) as temp:
        restored = restore_archive(archive, Path(temp) / "restored")
    if any(restored[k] != metadata[k] for k in ("files", "databases", "bytes")):
        raise ValueError("Состав восстановленной копии не совпадает")
    receipt = {
        "archive": str(archive),
        "sha256": metadata["sha256"],
        "files": restored["files"],
        "databases": restored["databases"],
        "bytes": restored["bytes"],
        "restoration_verified": True,
        "cloud_upload_verified": False,
    }
    archive.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    archive.with_suffix(".receipt.json").chmod(0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            capture_backup(args.host, args.remote_root, args.output), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
