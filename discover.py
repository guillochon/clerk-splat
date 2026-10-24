"""Find El Cajon Blvd reels: City Clerk archive index first, YouTube search as fallback."""
import re
import requests
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL
from .common import load_manifest, save_manifest, slug

YT_ID = re.compile(r"(?:youtu\.be/|v=|/embed/|/shorts/)([A-Za-z0-9_-]{11})")

def _matches(text, keywords):
    t = text.lower().replace(".", "")
    return any(k.replace(".", "") in t for k in keywords)

def from_archive(cfg):
    d = cfg["discover"]; found = {}; empty_streak = 0; prev_ids = None
    for page in range(d["archive_max_pages"]):
        r = requests.get(d["archive_url"], params={"page": page}, timeout=30)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a for a in soup.find_all("a", href=True) if YT_ID.search(a["href"])]
        # past the last page the site keeps serving the final page again
        ids = {YT_ID.search(a["href"]).group(1) for a in links}
        if ids and ids == prev_ids:
            break
        prev_ids = ids
        empty_streak = 0 if links else empty_streak + 1
        if empty_streak >= 3:
            break
        for a in links:
            # Title is usually the link text or its enclosing row/item
            ctx = a.get_text(" ", strip=True) or a.find_parent(["tr", "li", "div"]).get_text(" ", strip=True)
            if _matches(ctx, d["keywords"]):
                vid = YT_ID.search(a["href"]).group(1)
                found[vid] = ctx[:200]
        print(f"  archive page {page}: {len(found)} matches so far")
    return found

def from_youtube(cfg):
    d = cfg["discover"]
    q = f"ytsearch{d['youtube_search_limit']}:{d['youtube_search']}"
    with YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
        res = ydl.extract_info(q, download=False)
    return {e["id"]: e["title"] for e in res.get("entries", [])
            if e and _matches(e.get("title", ""), d["keywords"])}

def main(cfg):
    found = {}
    try:
        found = from_archive(cfg)
    except Exception as e:
        print(f"  archive scrape failed ({e}); falling back to YouTube search")
    if not found:
        found = from_youtube(cfg)
    items = load_manifest(cfg); known = {i["id"] for i in items}
    for vid, title in found.items():
        if vid not in known:
            items.append({"id": vid, "title": title, "segment": f"{slug(title)}_{vid}",
                          "url": f"https://www.youtube.com/watch?v={vid}", "include": True})
    save_manifest(cfg, items)
    print(f"{len(items)} reels in manifest. Review workspace/.../manifest.json and set include=false on any you don't want.")
