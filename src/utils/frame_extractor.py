import cv2
from typing import List, Tuple
from src.config import FRAME_INTERVAL_SECONDS


class FrameExtractor:
    def __init__(self, interval_seconds: float = FRAME_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds

    def extract(self, video_path: str) -> Tuple[List, List[float]]:
        """
        Read through the video at `video_path` and grab one frame every
        `interval_seconds`. Returns lists of (frame, timestamp).
        """
        print("\tExtracting frames...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.interval_seconds)

        frames, timestamps = [], []
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_interval == 0:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                frames.append(frame)
                timestamps.append(ts)
            frame_id += 1

        cap.release()
        return frames, timestamps
