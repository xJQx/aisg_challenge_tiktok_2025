from pathlib import Path
from yt_dlp import YoutubeDL
from src.config import VIDEO_DOWNLOAD_DIR

class VideoDownloader:
    def __init__(self, download_dir: Path = VIDEO_DOWNLOAD_DIR):
        self.download_dir = download_dir

    def download(self, youtube_url: str, qid: str, video_id: str) -> str:
        """
        Download a YouTube video (if not already present) and return its local path.
        """
        video_filename = f"{qid.split('-')[0]}_{video_id}.mp4"
        video_path = self.download_dir / video_filename
        if video_path.exists():
            print(f"\t✅ Already downloaded: {video_path}")
            return str(video_path)

        print(f"\t⬇️ Downloading video from YouTube: {youtube_url}")
        ydl_opts = {
            "format": "mp4",
            "outtmpl": str(video_path),
            "quiet": True,
            "noplaylist": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        return str(video_path)
