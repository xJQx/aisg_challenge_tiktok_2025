import json
from pathlib import Path
import cv2
from docarray import DocumentArray, dataclass, Document
from clip_client import Client
from datasets import Dataset, load_dataset

from src.config import OUTPUT_DIR, SKIP_PROCESSED_VIDEOS, ERROR_DIR
from src.utils.downloader import VideoDownloader
from src.utils.frame_extractor import FrameExtractor


records = []

# Define the document schema.
@dataclass
class Frame:
    ts: float


class FrameVectorizer:
    def __init__(
        self,
        server_url: str = 'grpc://0.0.0.0:51000',
        similarity_threshold: float = 0.5,
        output_dir: Path = OUTPUT_DIR
    ):
        # video downloader & frame extractor
        self.downloader = VideoDownloader()
        self.extractor = FrameExtractor()
        # Jina client for encoding
        self.client = Client(server=server_url)
        # similarity cutoff
        self.threshold = similarity_threshold
        # output directories
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ERROR_DIR.mkdir(parents=True, exist_ok=True)

    def process(self, example: dict):
        qid = example['qid']
        vid = example['video_id']
        duration = float(example['duration'])
        error_path = ERROR_DIR / f"{qid}_{vid}.json"
        try:
            question = example.get('question', '')
            # path to save embeddings as Parquet
            parquet_path = self.output_dir / f"{qid}_{vid}_embeddings.parquet"
            print(f"\nProcessing {qid}, video {vid}…")

            if SKIP_PROCESSED_VIDEOS and parquet_path.exists():
                print(f"\t✅ Already vectorized: {parquet_path}")
                return

            # 0. Download video
            video_path = self.downloader.download(
                example['youtube_url'], qid, vid
            )

            # 1. Extract frames and timestamps
            frames, timestamps = self.extractor.extract(video_path)

            # 2. Build DocumentArray for frames
            da = DocumentArray()
            for frame, ts in zip(frames, timestamps):
                success, buf = cv2.imencode('.jpg', frame)
                if not success:
                    continue
                doc = Document(blob=buf.tobytes())
                doc.tags['ts'] = ts
                da.append(doc)

            # 3. Vectorize frames via Jina encoder
            print("\tVectorizing frames via Jina encoder...")
            da = self.client.encode(da, show_progress=True)

            # 4. Vectorize question as a Document with text
            qn = self.client.encode([question])

            # 5. Find similar frames above threshold
            print(f"\tFinding {round(duration)} frames with the highest similarity")
            results = da.find(
                qn,
                limit=round(duration),  # limit to x matches, with x being the duration
                metric='cosine',
                threshold=self.threshold,
                show_progress=True
            )[0]

            # 6. Collect records from results
            records = []
            for match in results:
                sim = match.scores['cosine'].value
                #print("\tFrame:", match.tags['ts'], "Similarity:", sim)
                records.append({
                    'timestamp': match.tags['ts'],
                    'similarity': sim,
                    'embedding': match.embedding.tolist(),
                    'frame': match.blob
                })

            # 6. Save to Parquet via Hugging Face Dataset or write error
            if records:
                print(f"\tSaving {len(records)} similar frames to Parquet via HF Dataset...")
                ds = Dataset.from_list(records)
                ds.to_parquet(str(parquet_path))
                print(f"✓ Saved embeddings and frames to: {parquet_path}")
            else:
                # write an error file indicating no frames passed threshold
                error_msg = {
                    'qid': qid,
                    'video_id': vid,
                    'threshold': self.threshold,
                    'message': 'No frames passed the similarity threshold.'
                }
                with open(error_path, 'w') as ef:
                    json.dump(error_msg, ef, indent=2)
                print(f"⚠️ No frames passed threshold; wrote error to: {error_path}")
        except Exception as err:
            errors_msg = {
                'qid': qid,
                'video_id': vid,
                'error': str(err),
                'message': 'Error processing video.'
            }
            with open(error_path, "w") as f:
                json.dump(errors_msg, f, indent=2)


if __name__ == "__main__":
    # Iterate over the dataset
    server_url = 'grpc://0.0.0.0:51000'
    vectorizer = FrameVectorizer(server_url)
    dataset = load_dataset("lmms-lab/AISG_Challenge", split="test")
    for example in dataset:
        vectorizer.process(example)