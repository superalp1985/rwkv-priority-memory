"""Download and verify the pinned public XLM-R safetensors in resumable ranges."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from time import sleep

import requests

REV = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
URL = f"https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/{REV}/model.safetensors"
SIZE = 1115567652
SHA256 = "6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb"
CHUNK = 32 * 1024 * 1024


def verify(path):
    if path.stat().st_size != SIZE:
        raise ValueError("bad checkpoint size")
    with path.open("rb") as f:
        got = file_digest(f, "sha256").hexdigest()
    if got != SHA256:
        raise ValueError(f"bad checkpoint digest: {got}")


def fetch_chunk(directory, start):
    stop = min(start + CHUNK, SIZE) - 1
    path = directory / f"{start:010d}.chunk"
    count = stop - start + 1
    if path.exists():
        if path.stat().st_size != count:
            raise ValueError(f"incomplete existing chunk: {path}")
        return path
    for attempt in range(3):
        try:
            with requests.get(URL, params={"download": "true", "part": start}, stream=True,
                              timeout=(20, 40), headers={"Range": f"bytes={start}-{stop}"}) as r:
                if (r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{stop}/{SIZE}"):
                    raise OSError(f"range {start}-{stop}: HTTP {r.status_code}, {r.headers.get('Content-Range')}")
                with NamedTemporaryFile(dir=directory) as stage:
                    for block in r.iter_content(1024 * 1024):
                        stage.write(block)
                        if stage.tell() > count:
                            raise OSError("range exceeded requested size")
                    stage.flush()
                    if stage.tell() != count:
                        raise OSError("short range")
                    os.link(stage.name, path)
            return path
        except (OSError, requests.RequestException) as error:
            if attempt == 2:
                raise
            print(f"retry {start}: {type(error).__name__}", flush=True)
            sleep(attempt + 1)
    raise AssertionError("unreachable")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    dest = args.base / "model.safetensors"
    if not dest.exists():
        directory = args.base / "parts-32m"
        directory.mkdir(exist_ok=True)
        offsets = range(0, SIZE, CHUNK)
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = []
            for path in pool.map(lambda start: fetch_chunk(directory, start), offsets):
                paths.append(path)
                if len(paths) % 4 == 0 or len(paths) == len(offsets):
                    print(f"verified range lengths: {len(paths)}/{len(offsets)}", flush=True)
        with NamedTemporaryFile(dir=args.base) as stage:
            for path in paths:
                with path.open("rb") as f:
                    copyfileobj(f, stage, length=1024 * 1024)
            stage.flush()
            verify(Path(stage.name))
            os.link(stage.name, dest)
        for path in paths:
            path.unlink()
        directory.rmdir()
    verify(dest)
    print(f"{SHA256}  {dest}", flush=True)


if __name__ == "__main__":
    main()