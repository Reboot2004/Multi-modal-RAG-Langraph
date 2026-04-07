# ocr_engine.py
# ingestion/ocr_engine.py

import numpy as np
from PIL import Image
import io

from paddleocr import PaddleOCR
from config.settings import USE_GPU_FOR_OCR
from pipeline_logger import get_logger


logger = get_logger("ocr")


class OCREngine:
    def __init__(self):
        """
        Initialize PaddleOCR with multilingual support.
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",  # multilingual model auto-detects scripts
            use_gpu=USE_GPU_FOR_OCR,
        )
        logger.info("OCR engine initialized | use_gpu=%s", USE_GPU_FOR_OCR)

    def extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        """
        Extract text from raw image bytes (PNG/JPG).
        """

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        result = self.ocr.ocr(image_np)

        extracted_text = []

        if result:
            for line in result:
                if line:
                    for word_info in line:
                        text = word_info[1][0]
                        extracted_text.append(text)

        extracted = "\n".join(extracted_text)
        logger.debug("OCR extraction completed | words=%d | chars=%d", len(extracted_text), len(extracted))
        return extracted