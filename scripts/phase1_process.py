import cv2
from PIL import Image
from io import BytesIO
import base64
import json
from yt_dlp import YoutubeDL

from scripts.config import OUTPUT_DIR, FRAME_INTERVAL_SECONDS, VIDEO_DOWNLOAD_DIR, ERROR_DIR, SKIP_PROCESSED_VIDEOS

# Download YouTube Video
def _download_youtube_video(youtube_url, qid, video_id):
    video_download_path = VIDEO_DOWNLOAD_DIR / f"{qid.split('-')[0]}_{video_id}.mp4"
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
def _annotate_frames(frames, timestamps, main_question, call_model, question_id, video_id, batch_size=10):
    print("\tAnnotating Extracted Frames...")
    
    annotations = []
    for i in range(0, len(frames), batch_size):
        print(f"\t\tProcessing batch {i}...")
        batch_frames = frames[i:i + batch_size]
        batch_ts = timestamps[i:i + batch_size]
        batch_img_b64 = [__encode_frame_to_base64(f) for f in batch_frames]

        # Building Prompt
        previous_annotation = annotations[-1] if annotations else None
        prompt = f"""You are shown a series of {len(batch_frames)} image frames taken from a video. For each one,
        briefly describe what is happening in the image. Also describe any noticeable change from the previous frame.
        
        f"The user has asked: \"{main_question}\"\n"

        # Image Frame Timestamps
        """
        for j in range(len(batch_ts)):
            prompt += f"\nImage {j}: {batch_ts[j]:.2f} seconds into a video. Previous annotation: {previous_annotation}."
        prompt += """\n
        # Return Format (return nothing else. The response must be immediately parsable as an object in python, and the response must start with "[" and end with "]")
        [
            {
                "timestamp": 0.0,
                "annotation": ""
            },
            {
                "timestamp": 0.21,
                "annotation": ""
            }
            ...
        ]"""

        # Call Modal
        try:
            batch_annotations = call_model(prompt, batch_img_b64)

            # Parse response
            batch_annotations = batch_annotations.strip("```")
            batch_annotations = batch_annotations.strip("json")
            batch_annotations = batch_annotations.strip("```")
            if not batch_annotations.startswith("[") or not "".endswith("]"):
                l, r = 0, len(batch_annotations)-1
                while l < r:
                    if batch_annotations[l] != '[': l += 1
                    else: break
                while l < r:
                    if batch_annotations[r] != ']': r -= 1
                    else: break
                if l == r:
                    raise Exception("Invalid model response format!")
                batch_annotations = batch_annotations[l:r+1]

            batch_annotations = json.loads(batch_annotations)
            for a in batch_annotations:
                annotations.append(a)
        except Exception as err:
            with open(ERROR_DIR / f"{question_id}_{video_id}.json", "w") as f:
                print(batch_annotations)
                error_msg = {
                    "error_type": "annotate frames error",
                    "err": str(err),
                    "batch_annotations": str(batch_annotations)
                }
                json.dump(error_msg, f, indent=2)

    # for _, (frame, ts) in enumerate(zip(frames, timestamps)):
    #     img_b64 = __encode_frame_to_base64(frame)
        
    #     previous_annotation = annotations[-1] if annotations else None
    #     prompt = f"""Describe briefly what's happening in this image frame (base64 encoded JPEG) taken at {ts:.2f} seconds into a video.
    #     Note the changes (if any) from the previous frame. Here's the previous frame annotation: {previous_annotation}"""
        
    #     annotation = call_model(prompt, img_b64)
    #     annotations.append({
    #         "timestamp": ts,
    #         "annotation": annotation
    #     })

    return annotations

# Annotate Video as a whole
def _annotate_video_whole(main_question, frame_annotations, call_model):
    print("\tAnnotating video as a whole...")

    video_content = ""
    for i, frame_annotation in enumerate(frame_annotations):
        timestamp, annotation = frame_annotation["timestamp"], frame_annotation["annotation"]
        video_content += f"[Frame {i} (timestamp: {timestamp:.2f} seconds)] {annotation}\n"

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
    try:
        # Extract required fields
        video_id = example['video_id']
        question_id = example['qid']
        video_youtube_url = example["youtube_url"]
        main_question = example["question"]
        question_prompt = example["question_prompt"]

        print(f"\nProcessing question {question_id} video {video_id}...")

        # Skip if already processed
        if SKIP_PROCESSED_VIDEOS:
            output_file_path = OUTPUT_DIR / f"{question_id}_{video_id}.json"
            if output_file_path.exists():
                print(f"\t✅ Already processed: {output_file_path}. Skipping...")
                return

        # 0. Download Youtube Video to local if doesn't exist
        video_path = _download_youtube_video(video_youtube_url, question_id, video_id)

        # 1. Extract Frames
        frames, timestamps = _extract_frames(video_path, interval=FRAME_INTERVAL_SECONDS)

        # 2a. Annotate Extracted Frames
        frame_annotations = _annotate_frames(frames, timestamps, main_question, call_model, question_id, video_id)

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
    except Exception as err:
        with open(ERROR_DIR / f"{question_id}_{video_id}.json", "w") as f:
            json.dump(str(err), f, indent=2)
            
