#!/usr/bin/env python3
"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/.

Stdlib only, so it works before any dependency is installed.

Sources tried in order
  1. UCI static zip      https://archive.ics.uci.edu/static/public/45/heart+disease.zip
  2. UCI legacy file URL https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
  3. `ucimlrepo` package (only if already installed)
Or bypass the network:  python scripts/download_data.py --file /path/to/processed.cleveland.data

Manual fallback: download the zip from https://archive.ics.uci.edu/dataset/45/heart+disease,
extract `processed.cleveland.data` and put it in data/raw/.
"""

from __future__ import annotations

import argparse
import io
import shutil
import ssl
import sys
#import pandas
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "processed.cleveland.data"
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
FILE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
MEMBER = "processed.cleveland.data"
EXPECTED_ROWS, EXPECTED_COLS = 303, 14


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # present in the venv (dependency of mlflow/requests)

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    last: Exception | None = None
    ctx = _ssl_context()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "heartrisk-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310
                return resp.read()
        except Exception as exc:  # network errors are heterogeneous
            last = exc
            print(f"  attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")




#def fetch(url: str, retries: int = 3, timeout: int = 30) -> bytes:
#    last: Exception | None = None
#    for attempt in range(1, retries + 1):
#        try:
#            req = urllib.request.Request(url, headers={"User-Agent": "heartrisk-downloader/1.0"})
#            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
#                return resp.read()
##        except Exception as exc:  # network errors are heterogeneous
#            last = exc
#            print(f"  attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
#            time.sleep(2 * attempt)
#    raise RuntimeError(f"could not fetch {url}: {last}")


def from_zip() -> bytes:
    with zipfile.ZipFile(io.BytesIO(fetch(ZIP_URL))) as zf:
        return zf.read(MEMBER)


def from_file_url() -> bytes:
    return fetch(FILE_URL)


def from_ucimlrepo() -> bytes:
    from ucimlrepo import fetch_ucirepo  # type: ignore

    ds = fetch_ucirepo(id=45)
    frame = ds.data.features.copy()
    frame["num"] = ds.data.targets.iloc[:, 0].values
    return frame.to_csv(header=False, index=False, na_rep="?").encode()


def validate(raw: bytes) -> None:
    rows = [ln for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]
    bad = [r for r in rows if len(r.split(",")) != EXPECTED_COLS]
    if bad or len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"unexpected file shape: {len(rows)} rows (want {EXPECTED_ROWS}), "
            f"{len(bad)} malformed rows (want 0 with {EXPECTED_COLS} columns)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--file", help="use a local copy of processed.cleveland.data")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--prepare", action="store_true", help="also build data/processed/*.csv")
    args = ap.parse_args()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and not args.force and not args.file:
        print(f"[ok] {DEST} already exists (use --force to refresh)")
    else:
        if args.file:
            raw = Path(args.file).read_bytes()
        else:
            raw = None
            for name, getter in (("UCI zip", from_zip), ("UCI file", from_file_url),
                                 ("ucimlrepo", from_ucimlrepo)):  # fmt: skip
                try:
                    print(f"[..] trying {name}")
                    raw = getter()
                    validate(raw)
                    print(f"[ok] downloaded via {name}")
                    break
                except Exception as exc:
                    print(f"[x ] {name}: {exc}", file=sys.stderr)
                    raw = None
            if raw is None:
                print("\nAll sources failed. Download the dataset manually (see --help).",
                      file=sys.stderr)
                return 1
        validate(raw)
        tmp = DEST.with_suffix(".tmp")
        tmp.write_bytes(raw)
        shutil.move(str(tmp), DEST)
        print(f"[ok] saved {DEST} ({len(raw):,} bytes)")

    if args.prepare:
        sys.path.insert(0, str(ROOT / "src"))
        from heartrisk.data import prepare

        df = prepare()
        print(f"[ok] cleaned dataset: {len(df)} rows -> data/processed/heart_clean.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
