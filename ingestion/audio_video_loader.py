import os
import re
import time
from typing import Dict, List

from jiwer import cer, wer

from cache.av_cache import AVCache
from config.settings import (
    ASR_BEAM_SIZE,
    ASR_BENCHMARK_ENABLE_WER,
    ASR_CACHE_ENABLED,
    ASR_CHUNKING_STRATEGY,
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_ENABLE_QUALITY_FALLBACK,
    ASR_FALLBACK_LANGUAGES,
    ASR_FORCE_LANGUAGE,
    ASR_MAX_CHARS_PER_CHUNK,
    ASR_MAX_TRANSCRIBE_SECONDS,
    ASR_MIN_TEXT_QUALITY_SCORE,
    ASR_MODEL_SIZE,
    ASR_PROGRESS_LOG_EVERY_SEGMENTS,
    ASR_REDECODE_BEAM_SIZE,
    ASR_SEGMENT_GROUP_SIZE,
    ASR_SEGMENT_OVERLAP_SECONDS,
    ASR_SEGMENT_WINDOW_SECONDS,
    ASR_SUPPORTED_LANGUAGES,
    ASR_VAD_FILTER,
)
from ingestion.av_preprocessor import AVPreprocessor
from ingestion.av_segmenter import AVSegmenter
from pipeline_logger import get_logger
from processing.chunker import TextChunker
from processing.cleaner import TextCleaner
from processing.language_detector import LanguageDetector


logger = get_logger("audio_video_loader")


class AudioVideoLoader:
    AUDIO_VIDEO_EXTENSIONS = {
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
        ".mp4", ".mkv", ".mov", ".webm", ".avi",
    }

    def __init__(self):
        self.cleaner = TextCleaner()
        self.lang_detector = LanguageDetector()
        self.chunker = TextChunker()
        self.chunking_strategy = ASR_CHUNKING_STRATEGY
        self.segment_group_size = max(1, int(ASR_SEGMENT_GROUP_SIZE))
        self.max_chars_per_chunk = max(400, int(ASR_MAX_CHARS_PER_CHUNK))
        self.progress_every_segments = max(1, int(ASR_PROGRESS_LOG_EVERY_SEGMENTS))
        self.enable_quality_fallback = ASR_ENABLE_QUALITY_FALLBACK
        self.min_text_quality_score = float(ASR_MIN_TEXT_QUALITY_SCORE)
        self.fallback_languages = list(ASR_FALLBACK_LANGUAGES)
        self.force_language = ASR_FORCE_LANGUAGE
        self.cache_enabled = ASR_CACHE_ENABLED
        self.redecode_beam_size = max(1, int(ASR_REDECODE_BEAM_SIZE))

        self.preprocessor = AVPreprocessor()
        self.segmenter = AVSegmenter(
            window_seconds=ASR_SEGMENT_WINDOW_SECONDS,
            overlap_seconds=ASR_SEGMENT_OVERLAP_SECONDS,
        )
        self.cache = AVCache()

        self._whisper_import_error = None
        whisper_cls = None
        try:
            from faster_whisper import WhisperModel as _WhisperModel

            whisper_cls = _WhisperModel
        except Exception as ex:
            self._whisper_import_error = ex

        if whisper_cls is None:
            logger.error("Failed to import faster-whisper backend: %s", self._whisper_import_error)
            self.model = None
            return

        self.model = whisper_cls(
            ASR_MODEL_SIZE,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        )
        logger.info(
            "AudioVideoLoader initialized | model=%s | device=%s | compute_type=%s | chunking=%s | cache=%s",
            ASR_MODEL_SIZE,
            ASR_DEVICE,
            ASR_COMPUTE_TYPE,
            self.chunking_strategy,
            self.cache_enabled,
        )

    def _text_quality_score(self, text: str) -> float:
        if not text:
            return 0.0

        alnum_chars = [ch for ch in text if ch.isalpha() or ch.isdigit()]
        total = len(alnum_chars)
        if total == 0:
            return 0.0

        words = re.findall(r"\w+", text, flags=re.UNICODE)
        if not words:
            return 0.0

        unique_words = len({w.lower() for w in words})
        unique_word_ratio = unique_words / max(1, len(words))

        unique_chars = len(set(alnum_chars))
        unique_char_ratio = unique_chars / max(1, total)

        avg_word_len = sum(len(w) for w in words) / max(1, len(words))
        longest_repeat = max((len(m.group(0)) for m in re.finditer(r"(.)\1+", text)), default=1)

        freq = {}
        for w in words:
            key = w.lower()
            freq[key] = freq.get(key, 0) + 1
        top_word_ratio = max(freq.values()) / max(1, len(words))

        score = 0.0
        score += 0.35 * min(1.0, unique_word_ratio * 2.0)
        score += 0.35 * min(1.0, unique_char_ratio * 4.0)
        if 2.0 <= avg_word_len <= 12.0:
            score += 0.2
        elif avg_word_len <= 20.0:
            score += 0.08
        score += 0.1 * min(1.0, len(words) / 80.0)

        if longest_repeat >= 6:
            score -= 0.30
        if top_word_ratio > 0.5:
            score -= 0.25
        if unique_char_ratio < 0.08:
            score -= 0.35
        if len(words) >= 8 and unique_words <= 2:
            score -= 0.25

        return max(0.0, min(1.0, score))

    def _is_bad_segment(self, text: str, score: float) -> bool:
        if not text or len(text.strip()) < 3:
            return True
        if score < (self.min_text_quality_score * 0.7):
            return True
        if re.search(r"(.)\1{8,}", text):
            return True
        return False

    def _decode_segment(self, file_path: str, start_sec: float, end_sec: float, language: str, vad_filter: bool, beam_size: int):
        segments, _ = self.model.transcribe(
            file_path,
            beam_size=beam_size,
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=vad_filter,
            language=language,
            clip_timestamps=[start_sec, end_sec],
        )

        text_parts = []
        avg_logs = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                text_parts.append(t)
                avg_logs.append(float(getattr(seg, "avg_logprob", 0.0) or 0.0))

        text = self.cleaner.clean(" ".join(text_parts))
        quality = self._text_quality_score(text)
        avg_logprob = sum(avg_logs) / len(avg_logs) if avg_logs else 0.0

        return {
            "text": text,
            "quality": quality,
            "avg_logprob": avg_logprob,
        }

    def _decode_all_segments(self, file_path: str, segments: List[Dict], language: str, vad_filter: bool, beam_size: int, source_name: str):
        rows = []
        t_start = time.perf_counter()
        total = len(segments)

        for i, seg in enumerate(segments, start=1):
            start = float(seg["start"])
            end = float(seg["end"])
            decoded = self._decode_segment(
                file_path=file_path,
                start_sec=start,
                end_sec=end,
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
            )

            rows.append(
                {
                    "idx": int(seg["idx"]),
                    "start": start,
                    "end": end,
                    "text": decoded["text"],
                    "avg_logprob": float(decoded["avg_logprob"]),
                    "quality": float(decoded["quality"]),
                }
            )

            if i == 1 or i % self.progress_every_segments == 0 or i == total:
                elapsed = time.perf_counter() - t_start
                pct = (i / max(1, total)) * 100.0
                eta = (elapsed / max(1e-6, i)) * (total - i)
                logger.info(
                    "ASR progress | source=%s | windows=%d/%d | progress=%.1f%% | elapsed=%.1fs | eta≈%.1fs",
                    source_name,
                    i,
                    total,
                    pct,
                    elapsed,
                    eta,
                )

        return rows

    def _rows_to_transcript(self, rows: List[Dict]) -> str:
        return self.cleaner.clean("\n".join([r.get("text", "") for r in rows if r.get("text", "").strip()]))

    def _build_segment_grouped_chunks(self, segment_rows: List[Dict], source: str, language: str) -> List[Dict]:
        chunks = []
        buffer_rows = []

        def flush_buffer():
            if not buffer_rows:
                return
            text = self.cleaner.clean(" ".join([r.get("text", "") for r in buffer_rows]))
            if not text:
                return
            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "page": 1,
                    "language": language,
                    "start_sec": float(buffer_rows[0]["start"]),
                    "end_sec": float(buffer_rows[-1]["end"]),
                }
            )

        for row in segment_rows:
            if not row.get("text", "").strip():
                continue
            buffer_rows.append(row)
            char_count = sum(len(x.get("text", "")) for x in buffer_rows)
            if len(buffer_rows) >= self.segment_group_size or char_count >= self.max_chars_per_chunk:
                flush_buffer()
                buffer_rows = []

        if buffer_rows:
            flush_buffer()

        return chunks

    def parse(self, file_path: str, source_name: str = None, reference_text: str = None) -> Dict:
        if self.model is None:
            raise RuntimeError(
                "Audio/video ASR backend is unavailable. "
                f"faster-whisper import failed: {self._whisper_import_error}"
            )

        file_name = source_name or os.path.basename(file_path)
        extension = os.path.splitext(file_name)[1].lower()
        if extension not in self.AUDIO_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported audio/video type: {extension}")

        t0 = time.perf_counter()
        media_hash = self.cache.media_hash(file_path)

        preprocess_stage = self.cache.load_stage(media_hash, "preprocess_v2") if self.cache_enabled else None
        if preprocess_stage is None:
            preprocess_stage = self.preprocessor.preprocess(file_path, source_name=file_name)
            if self.cache_enabled:
                self.cache.save_stage(media_hash, "preprocess_v2", preprocess_stage)

        duration_seconds = float(preprocess_stage.get("duration_seconds", 0.0) or 0.0)
        effective_duration = min(duration_seconds, float(ASR_MAX_TRANSCRIBE_SECONDS)) if ASR_MAX_TRANSCRIBE_SECONDS and ASR_MAX_TRANSCRIBE_SECONDS > 0 else duration_seconds

        segment_stage = self.cache.load_stage(media_hash, "segments_v2") if self.cache_enabled else None
        if segment_stage is None:
            segments = self.segmenter.build_segments(duration_seconds=duration_seconds, max_seconds=ASR_MAX_TRANSCRIBE_SECONDS)
            segment_stage = {"segments": segments}
            if self.cache_enabled:
                self.cache.save_stage(media_hash, "segments_v2", segment_stage)
        segments = segment_stage.get("segments", [])

        final_stage = self.cache.load_stage(media_hash, "asr_final_v2") if self.cache_enabled else None
        if final_stage is not None:
            segment_rows = final_stage.get("segment_rows", [])
            cleaned = final_stage.get("transcript", "")
            detected_language = final_stage.get("detected_language", "en")
            decode_seconds = float(final_stage.get("decode_seconds", 0.0))
            quality_score = float(final_stage.get("quality_score", 0.0))
            logger.info("ASR cache hit | source=%s | segments=%d", file_name, len(segment_rows))
        else:
            decode_start = time.perf_counter()
            seed_language = self.force_language if self.force_language else None

            segment_rows = self._decode_all_segments(
                file_path=file_path,
                segments=segments,
                language=seed_language,
                vad_filter=ASR_VAD_FILTER,
                beam_size=ASR_BEAM_SIZE,
                source_name=file_name,
            )
            cleaned = self._rows_to_transcript(segment_rows)
            quality_score = self._text_quality_score(cleaned)
            detected_language = self.lang_detector.detect_language(cleaned)

            logger.info(
                "ASR pass[auto] | source=%s | language=%s | quality=%.3f | chars=%d",
                file_name,
                detected_language,
                quality_score,
                len(cleaned),
            )

            if self.enable_quality_fallback:
                bad_indexes = [i for i, row in enumerate(segment_rows) if self._is_bad_segment(row.get("text", ""), float(row.get("quality", 0.0)))]
                if bad_indexes:
                    logger.warning(
                        "ASR fallback triggered | source=%s | bad_windows=%d/%d",
                        file_name,
                        len(bad_indexes),
                        len(segment_rows),
                    )
                    for idx in bad_indexes:
                        seg = segment_rows[idx]
                        best = seg
                        best_score = float(seg.get("quality", 0.0))
                        for lang in self.fallback_languages:
                            if seed_language and lang == seed_language:
                                continue
                            candidate = self._decode_segment(
                                file_path=file_path,
                                start_sec=float(seg["start"]),
                                end_sec=float(seg["end"]),
                                language=lang,
                                vad_filter=False,
                                beam_size=self.redecode_beam_size,
                            )
                            if float(candidate["quality"]) > best_score:
                                best = {
                                    "idx": seg["idx"],
                                    "start": seg["start"],
                                    "end": seg["end"],
                                    "text": candidate["text"],
                                    "avg_logprob": float(candidate["avg_logprob"]),
                                    "quality": float(candidate["quality"]),
                                }
                                best_score = float(candidate["quality"])

                            if best_score >= self.min_text_quality_score:
                                break

                        segment_rows[idx] = best

                    cleaned = self._rows_to_transcript(segment_rows)
                    quality_score = self._text_quality_score(cleaned)
                    detected_language = self.lang_detector.detect_language(cleaned)

            decode_seconds = time.perf_counter() - decode_start
            if self.cache_enabled:
                self.cache.save_stage(
                    media_hash,
                    "asr_final_v2",
                    {
                        "segment_rows": segment_rows,
                        "transcript": cleaned,
                        "detected_language": detected_language,
                        "decode_seconds": round(decode_seconds, 3),
                        "quality_score": round(quality_score, 4),
                    },
                )

        if not cleaned.strip():
            return {
                "chunks": [],
                "debug": {
                    "source": file_name,
                    "detected_language": detected_language,
                    "segments": [],
                    "transcript": "",
                    "benchmark": {},
                },
            }

        if detected_language not in ASR_SUPPORTED_LANGUAGES:
            detected_language = self.lang_detector.detect_language(cleaned)

        chunk_start = time.perf_counter()
        if self.chunking_strategy == "segment_grouped":
            chunks = self._build_segment_grouped_chunks(segment_rows, file_name, detected_language)
        elif self.chunking_strategy == "hybrid":
            grouped = self._build_segment_grouped_chunks(segment_rows, file_name, detected_language)
            chunks = []
            for chunk in grouped:
                if len(chunk.get("text", "")) <= self.max_chars_per_chunk:
                    chunks.append(chunk)
                else:
                    chunks.extend(
                        self.chunker.chunk_text(
                            text=chunk.get("text", ""),
                            source=file_name,
                            page=1,
                            language=detected_language,
                        )
                    )
        else:
            chunks = self.chunker.chunk_text(
                text=cleaned,
                source=file_name,
                page=1,
                language=detected_language,
            )
        chunk_seconds = time.perf_counter() - chunk_start

        total_seconds = time.perf_counter() - t0
        benchmark = {
            "decode_seconds": round(float(decode_seconds), 3),
            "chunking_seconds": round(chunk_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "audio_seconds": round(float(effective_duration), 3),
            "rtf": round(float(decode_seconds) / max(1e-6, float(effective_duration)), 4),
            "segments": len(segment_rows),
            "chunks": len(chunks),
            "chars": len(cleaned),
            "quality_score": round(float(quality_score), 4),
            "chunking_strategy": self.chunking_strategy,
            "cache_enabled": self.cache_enabled,
            "max_transcribe_seconds": int(ASR_MAX_TRANSCRIBE_SECONDS or 0),
        }

        if reference_text and ASR_BENCHMARK_ENABLE_WER:
            ref_clean = self.cleaner.clean(reference_text)
            if ref_clean.strip():
                benchmark["wer"] = round(float(wer(ref_clean, cleaned)), 4)
                benchmark["cer"] = round(float(cer(ref_clean, cleaned)), 4)

        logger.info(
            "ASR completed | source=%s | detected_language=%s | segments=%d | chunks=%d | decode_seconds=%.2f | chunking_seconds=%.2f | total=%.2f | rtf=%.3f",
            file_name,
            detected_language,
            len(segment_rows),
            len(chunks),
            float(decode_seconds),
            chunk_seconds,
            total_seconds,
            benchmark["rtf"],
        )

        for row in segment_rows[:15]:
            logger.debug(
                "ASR segment | source=%s | idx=%d | start=%.2f | end=%.2f | quality=%.3f | text=%s",
                file_name,
                row["idx"],
                row["start"],
                row["end"],
                float(row.get("quality", 0.0)),
                row.get("text", ""),
            )

        return {
            "chunks": chunks,
            "debug": {
                "source": file_name,
                "detected_language": detected_language,
                "segments": segment_rows,
                "transcript": cleaned,
                "benchmark": benchmark,
            },
        }
