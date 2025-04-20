import cv2
from PIL import Image
from io import BytesIO
import base64
import json
from yt_dlp import YoutubeDL

from scripts.config import OUTPUT_DIR, FRAME_INTERVAL_SECONDS, VIDEO_DOWNLOAD_DIR

# Download YouTube Video
def _download_youtube_video(youtube_url, qid, video_id):
    video_download_path = VIDEO_DOWNLOAD_DIR / f"{qid}_{video_id}.mp4"
    if video_download_path.exists():
        print(f"\t✅ Already downloaded: {video_download_path}")
        return str(video_download_path)

    print(f"\t⬇️ Downloading video from YouTube: {youtube_url}")
    ydl_opts = {
        "format": "mp4",
        "outtmpl": str(video_download_path),
        "quiet": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return str(video_download_path)

# Extract frames every N seconds
def _extract_frames(video_path, interval):
    print("\tExtracting frames...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frames = []
    timestamps = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frames.append(frame)
            timestamps.append(timestamp)
        frame_id += 1
    cap.release()
    return frames, timestamps

# Convert frame to base64 image string
def __encode_frame_to_base64(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Annotate Frames
def _annotate_frames(frames, timestamps, call_model):
    print("\tAnnotating Extracted Frames...")
    
    annotations = []
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        img_b64 = __encode_frame_to_base64(frame)
        prompt = f"Describe briefly what's happening in this image frame (base64 encoded JPEG) taken at {ts:.2f} seconds into a video: {img_b64}\n"
        annotation = call_model(prompt)
        annotations.append({
            "timestamp": ts,
            "annotation": annotation
        })

    return annotations

# Annotate Video as a whole
def _annotate_video_whole(main_question, frame_annotations, call_model):
    print("\tAnnotating video as a whole...")

    video_content = ""
    for i, frame_annotation in enumerate(frame_annotations):
        timestamp, annotation = frame_annotation["timestamp"], frame_annotation["annotation"]
        video_content += f"[Frame {i} (timestamp: {timestamp:.2f} seconds)] {annotation}\n"
    
    # video_content = ""
    # for i, (frame, ts) in enumerate(zip(frames, timestamps)):
    #     img_b64 = __encode_frame_to_base64(frame)
    #     video_content += f"[Frame {i} (timestamp: {ts:.2f} seconds)] Image: {img_b64}\n"

    prompt = \
        f"You are analyzing a video with the following content (Frames snapshots with annotations)\n" +\
        video_content + "\n" +\
        f"The user has asked: \"{main_question}\"\n" +\
        f"Provide a short general description of what is happening in the video."

    annotation = call_model(prompt)
    return annotation

# Summarize Annotations
def _summarize_all_annotations(frame_annotations: list, whole_video_annotation, call_model) -> str:
    print("\tSummarizing all annotations...")

    annotation_text = "\n".join(
        [f"[{round(a['timestamp'], 1)}s]: {a['annotation']}" for a in frame_annotations]
    )

    prompt = (
        f"You are given frame-level descriptions of a video:\n"
        f"{annotation_text}\n\n"
        f"You are also given video-level description of the same video:\n"
        f"{whole_video_annotation}"
        f"Using the provided information, provide a brief summary of the descriptions."
    )
    summary = call_model(prompt)
    return summary

# Generate sub-questions from main question
def _generate_sub_questions(main_question: str, call_model):
    print("\tGenerating sub-questions from main question...")
    subq_prompt = f"Given the main question: '{main_question}', generate 3 helpful sub-questions to better understand the video."
    sub_questions = call_model(subq_prompt)
    return sub_questions

# Main logic
def phase1_process_video(example, call_model):
    # Extract required fields
    video_id = example['video_id']
    question_id = example['qid']
    video_youtube_url = example["youtube_url"]
    main_question = example["question"]
    question_prompt = example["question_prompt"]

    print(f"\nProcessing question {question_id} video {video_id}...")

    # 0. Download Youtube Video to local if doesn't exist
    video_path = _download_youtube_video(video_youtube_url, question_id, video_id)

    # 1. Extract Frames
    frames, timestamps = _extract_frames(video_path, interval=FRAME_INTERVAL_SECONDS)

    # 2a. Annotate Extracted Frames
    frame_annotations = _annotate_frames(frames, timestamps, call_model)

    # 2b.Annotate Video as a Whole
    whole_video_annotation = _annotate_video_whole(main_question, frame_annotations, call_model)

    # 3. Summarise all annotations
    annotations_summary = _summarize_all_annotations(frame_annotations, whole_video_annotation, call_model)

    # 4. Generate sub-questions from main question
    sub_questions = _generate_sub_questions(main_question, call_model)

    # Save results
    output = {
        "qid": question_id,
        "video_id": video_id,
        "video_path": video_path,
        "main_question": main_question,
        "sub_questions": sub_questions,
        "annotations": {
            "frame_annotations": frame_annotations,
            "whole_video_annotation": whole_video_annotation,
            "annotations_summary": annotations_summary
        }
    }
    with open(OUTPUT_DIR / f"{question_id}_{video_id}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Done. Annotations saved for question {question_id} video {video_id}.")
