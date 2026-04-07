import json
import os
from pathlib import Path
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "data" / "test_media"

CATEGORIES = {
    "audio_hi": "Category:Hindi_pronunciation",
    "audio_ta": "Category:Tamil_pronunciation",
    "audio_te": "Category:Telugu_pronunciation",
    "audio_bn": "Category:Bengali_pronunciation",
    "video_hi": "Category:Videos_in_Hindi",
    "video_ta": "Category:Videos_in_Tamil",
}

ALLOWED_EXTS = {".ogg", ".oga", ".wav", ".mp3", ".webm", ".ogv", ".mp4"}
HEADERS = {
    "User-Agent": "indic-multimodal-rag-test-media-fetcher/1.0 (educational testing)",
}


def list_category_titles(category: str, limit: int = 30):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": limit,
        "format": "json",
    }
    r = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    members = r.json().get("query", {}).get("categorymembers", [])
    return [m.get("title", "") for m in members if m.get("title", "").startswith("File:")]


def commons_direct_url(file_title: str):
    filename = file_title.replace("File:", "", 1)
    encoded = quote(filename, safe="")
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}"


def pick_and_download(group: str, category: str):
    titles = list_category_titles(category)
    group_dir = OUT_DIR / group
    group_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for title in titles:
        candidate_name = title.replace("File:", "", 1)
        ext = os.path.splitext(candidate_name.lower())[1]
        if ext not in ALLOWED_EXTS:
            continue

        url = commons_direct_url(title)
        target = group_dir / candidate_name

        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            if target.stat().st_size < 2048:
                target.unlink(missing_ok=True)
                continue

            downloaded.append(
                {
                    "title": title,
                    "url": url,
                    "path": str(target.relative_to(BASE_DIR)),
                    "bytes": target.stat().st_size,
                }
            )
            break
        except Exception:
            continue

    return downloaded


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"items": []}

    for group, category in CATEGORIES.items():
        items = pick_and_download(group, category)
        manifest["items"].extend(items)

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved manifest: {manifest_path}")
    print(f"Downloaded items: {len(manifest['items'])}")
    for item in manifest["items"]:
        print(f"- {item['path']} ({item['bytes']} bytes)")


if __name__ == "__main__":
    main()
