#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch upload PDFs to OpenAI Files API and persist mapping to JSON.

Features:
- Scan a directory (default: ./pdfs) for .pdf
- Load an existing mapping JSON (default: ./uploaded_files.json)
- Skip PDFs already uploaded (by SHA256 hash OR by relative path if hash missing)
- Upload only new PDFs
- Append new records and write back to JSON

Usage:
  python upload_pdfs.py
  python upload_pdfs.py --pdf_dir ./pdfs --map_json ./uploaded_files.json --purpose user_data
"""

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv("./.env")

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_mapping(map_path: Path) -> Dict[str, Any]:
    if not map_path.exists():
        return {
            "meta": {
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "format_version": "1.0",
            },
            "files": []  # list of records
        }
    with map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "files" not in data or not isinstance(data["files"], list):
        raise ValueError(f"Invalid mapping JSON format: {map_path}")
    return data


def save_mapping(map_path: Path, data: Dict[str, Any]) -> None:
    data["meta"]["updated_at"] = now_iso()
    tmp = map_path.with_suffix(map_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(map_path)


def index_existing(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build indices for fast lookup:
    - by_sha256
    - by_relpath
    """
    by_sha = {}
    by_path = {}
    for rec in data["files"]:
        sha = rec.get("sha256")
        rel = rec.get("relpath")
        if sha:
            by_sha[sha] = rec
        if rel:
            by_path[rel] = rec
    return {"by_sha256": by_sha, "by_relpath": by_path}


def iter_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./pdfs", help="Directory containing PDFs")
    parser.add_argument("--map_json", default="./uploaded_files.json", help="Output mapping JSON path")
    parser.add_argument("--purpose", default="user_data", help="OpenAI file purpose (e.g. user_data, assistants)")
    parser.add_argument("--dry_run", action="store_true", help="Scan and compute hashes but do not upload")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    pdf_dir = Path(args.pdf_dir).resolve()
    map_path = Path(args.map_json).resolve()

    if not pdf_dir.exists():
        raise FileNotFoundError(f"pdf_dir does not exist: {pdf_dir}")

    client = OpenAI(api_key=api_key)

    data = load_mapping(map_path)
    idx = index_existing(data)

    pdfs = iter_pdfs(pdf_dir)
    print(f"Scanning PDFs in: {pdf_dir}")
    print(f"Found {len(pdfs)} PDF(s).")
    print(f"Mapping file: {map_path} (existing records: {len(data['files'])})")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPLOAD'}")
    print("-" * 60)

    uploaded_count = 0
    skipped_count = 0
    error_count = 0

    for p in pdfs:
        relpath = str(p.relative_to(pdf_dir)).replace("\\", "/")
        size = p.stat().st_size
        mtime = int(p.stat().st_mtime)

        # Compute SHA256 to avoid duplicate uploads even if filename changes
        try:
            sha = sha256_file(p)
        except Exception as e:
            print(f"[ERROR] hash failed: {relpath} -> {e}")
            error_count += 1
            continue

        # Skip if already uploaded (prefer sha256 match)
        if sha in idx["by_sha256"]:
            rec = idx["by_sha256"][sha]
            print(f"[SKIP] already uploaded (sha256 match): {relpath} -> {rec.get('file_id')}")
            skipped_count += 1
            # Optional: update relpath if missing
            if not rec.get("relpath"):
                rec["relpath"] = relpath
                idx["by_relpath"][relpath] = rec
            continue

        # Secondary skip check: same relpath
        if relpath in idx["by_relpath"]:
            rec = idx["by_relpath"][relpath]
            print(f"[SKIP] already in mapping (relpath match): {relpath} -> {rec.get('file_id')}")
            skipped_count += 1
            # Optional: if sha missing in old record, backfill
            if not rec.get("sha256"):
                rec["sha256"] = sha
                idx["by_sha256"][sha] = rec
            continue

        print(f"[NEW ] {relpath} (size={size} bytes, sha256={sha[:12]}...)")

        if args.dry_run:
            # Record as "planned" without uploading (optional)
            continue

        # Upload
        try:
            with p.open("rb") as f:
                uploaded = client.files.create(
                    file=f,
                    purpose=args.purpose
                )
            file_id = uploaded.id
            print(f"      uploaded -> {file_id}")

            rec = {
                "relpath": relpath,
                "abspath": str(p),
                "filename": p.name,
                "sha256": sha,
                "size_bytes": size,
                "mtime_unix": mtime,
                "purpose": args.purpose,
                "file_id": file_id,
                "uploaded_at": now_iso(),
            }

            data["files"].append(rec)
            # Update indices
            idx["by_sha256"][sha] = rec
            idx["by_relpath"][relpath] = rec

            uploaded_count += 1

            # Save after each upload (crash-safe)
            save_mapping(map_path, data)

        except Exception as e:
            print(f"[ERROR] upload failed: {relpath} -> {e}")
            error_count += 1
            continue

    print("-" * 60)
    print(f"Done. uploaded={uploaded_count}, skipped={skipped_count}, errors={error_count}")
    print(f"Mapping saved to: {map_path}")


if __name__ == "__main__":
    main()
