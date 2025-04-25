import base64
from io import BytesIO
from PIL import Image
import cv2


def encode_to_base64(frame) -> str:
    """
    Convert an OpenCV BGR frame to a base64-encoded JPEG.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def encode_blob_to_base64(frame_blob: bytes) -> str:
    return base64.b64encode(frame_blob).decode("utf-8")