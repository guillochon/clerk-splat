from yt_dlp import YoutubeDL
from .common import load_manifest

def main(cfg):
    raw = cfg["workspace"] / "raw"; raw.mkdir(exist_ok=True)
    for item in [i for i in load_manifest(cfg) if i.get("include", True)]:
        out = raw / f"{item['segment']}.%(ext)s"
        if list(raw.glob(f"{item['segment']}.*")):
            continue
        opts = {"format": cfg["download"]["format"], "outtmpl": str(out),
                "writeinfojson": True, "quiet": False}
        with YoutubeDL(opts) as ydl:
            ydl.download([item["url"]])
