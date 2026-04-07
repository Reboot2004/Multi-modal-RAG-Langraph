import os
from typing import Dict

import av

from pipeline_logger import get_logger


logger = get_logger("av_preprocessor")


class AVPreprocessor:
    def preprocess(self, file_path: str, source_name: str = None) -> Dict:
        file_name = source_name or os.path.basename(file_path)

        duration_seconds = 0.0
        sample_rate = 0
        channels = 0

        container = av.open(file_path)
        try:
            if container.duration is not None:
                duration_seconds = float(container.duration) / 1_000_000.0

            for stream in container.streams:
                if stream.type == "audio":
                    sample_rate = int(getattr(stream.codec_context, "sample_rate", 0) or 0)
                    channels = int(getattr(stream.codec_context, "channels", 0) or 0)
                    break
        finally:
            container.close()

        metadata = {
            "source": file_name,
            "path": file_path,
            "duration_seconds": round(duration_seconds, 3),
            "sample_rate": sample_rate,
            "channels": channels,
        }
        logger.info(
            "AV preprocess | source=%s | duration=%.2fs | sample_rate=%s | channels=%s",
            file_name,
            duration_seconds,
            sample_rate,
            channels,
        )
        return metadata
