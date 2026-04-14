# settings.py
# config/settings.py

import os

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except Exception:
    # Keep running even if python-dotenv is not installed.
    pass

# ==============================
# PROJECT ROOT
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# ==============================
# GROQ CONFIGURATION
# ==============================

# Set your Groq API key as environment variable:
# export GROQ_API_KEY="your_key_here"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# OpenRouter key for optional provider toggle
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Provider defaults
LLM_PROVIDER_DEFAULT = "groq"  # groq | openrouter

# Model defaults per provider
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
OPENROUTER_MODEL_NAME = "nvidia/nemotron-nano-9b-v2:free"

# Backward compatibility alias
LLM_MODEL_NAME = GROQ_MODEL_NAME

# ==============================
# EMBEDDING MODEL CONFIG
# ==============================

# Multilingual embedding model (strong for Indic + English)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Embedding dimension for bge-m3 = 1024
EMBEDDING_DIMENSION = 1024

# Embedding runtime tuning (useful for CPU/8GB environments)
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_MAX_SEQ_LENGTH = 384
EMBEDDING_MAX_WORDS_PER_CHUNK = 300

# Fast profile overrides for large PDFs
FAST_CHUNK_SIZE = 700
FAST_CHUNK_OVERLAP = 100
FAST_EMBEDDING_MAX_WORDS_PER_CHUNK = 250

# ==============================
# CHUNKING CONFIG
# ==============================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 80

# Chunking backend
# - "chonkie": use Chonkie RecursiveChunker with overlap
# - "legacy": use built-in chunker fallback
CHUNKER_BACKEND = "chonkie"

# ==============================
# RETRIEVAL CONFIG
# ==============================

TOP_K = 5
ENABLE_HYBRID_DB = True
LEXICAL_TOP_K = 20
LEXICAL_DB_PATH = os.path.join(FAISS_INDEX_DIR, "lexical_index.db")

# Weighted score fusion for semantic + HyPE + lexical retrieval
LEXICAL_SCORE_WEIGHT = 0.2

# ==============================
# LANGGRAPH QUERY ORCHESTRATION
# ==============================

LANGGRAPH_ENABLE_QUERY_EXPANSION = True
LANGGRAPH_EXPANSION_VARIANTS = 3
LANGGRAPH_INITIAL_TOP_K = 20
LANGGRAPH_FUSION_RRF_K = 60
LANGGRAPH_MAX_CHUNKS_PER_SOURCE = 3
LANGGRAPH_FINAL_TOP_K = 5

# ==============================
# DEBUGGING
# ==============================

PIPELINE_DEBUG = True

# ==============================
# HYPE (Hypothetical Prompt Embeddings)
# ==============================

ENABLE_HYPE = True
HYPE_PROMPTS_PER_CHUNK = 2
HYPE_QUERY_TOP_K = 12
HYPE_SOURCE_CHAR_LIMIT = 1200
HYPE_MAX_CHUNKS_PER_DOCUMENT = 12
FAST_HYPE_PROMPTS_PER_CHUNK = 1
ENABLE_HYPE_CACHE = True
HYPE_CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "hype_prompt_cache.json")

# Weighted score fusion: final = (semantic * w1) + (hype * w2)
SEMANTIC_SCORE_WEIGHT = 0.6
HYPE_SCORE_WEIGHT = 0.4

# ==============================
# SELF-RAG (Quality Control Gates)
# ==============================

ENABLE_SELF_RAG = True  # Enable quality gates on all answers

# Gate thresholds (0.0 to 1.0)
SELF_RAG_DOC_RELEVANCE_THRESHOLD = 0.5  # Below this: log warning, don't block
SELF_RAG_FAITHFULNESS_THRESHOLD = 0.6   # Below this: affects confidence badge
SELF_RAG_USEFULNESS_THRESHOLD = 0.5     # Below this: affects confidence badge

# Hard refusal (don't answer if confidence below threshold)
SELF_RAG_HARD_REFUSAL_ENABLED = False  # Set to True for mission-critical apps
SELF_RAG_HARD_REFUSAL_THRESHOLD = 0.4   # Refuse if overall confidence < this

# Confidence score weights
SELF_RAG_WEIGHT_RETRIEVAL = 0.1
SELF_RAG_WEIGHT_RELEVANCE = 0.1
SELF_RAG_WEIGHT_FAITHFULNESS = 0.4
SELF_RAG_WEIGHT_USEFULNESS = 0.4

# ==============================
# CONTEXTUAL RETRIEVAL (Multi-turn Conversation)
# ==============================

ENABLE_CONTEXTUAL_RETRIEVAL = True  # Rewrite queries using chat history
CONTEXT_HISTORY_TURNS = 3  # Use last N turns for contextualization
MAX_CONTEXT_HISTORY_CHARS = 2000  # Max chars of history to send to LLM

# ==============================
# CONVERSATION MEMORY
# ==============================

ENABLE_CONVERSATION_MEMORY = True
CONVERSATION_HISTORY_LIMIT = 20  # Store last N Q&A pairs
MEMORY_STORE_PATH = os.path.join(PROCESSED_DATA_DIR, "conversation_memory.json")

# ==============================
# TIER 1: Adaptive Retrieval
# ==============================

ENABLE_ADAPTIVE_RETRIEVAL = True  # Adjust k based on query complexity
ADAPTIVE_SIMPLE_QUERY_K_OFFSET = -2  # Reduce k for simple queries
ADAPTIVE_COMPLEX_QUERY_K_OFFSET = 3  # Increase k for complex queries

# ==============================
# TIER 1: LLM-based Re-ranking
# ==============================

ENABLE_LLM_RERANKING = True  # Use Groq to re-score retrieved chunks
LLM_RERANK_TOP_K = 5  # Re-rank top 5 candidates
LLM_RERANK_BATCH_SIZE = 5  # Process in batches

# ==============================
# TIER 2: Fallback Retrieval
# ==============================

ENABLE_FALLBACK_RETRIEVAL = True  # Graceful degradation strategies
FALLBACK_MIN_RESULTS = 2  # Trigger fallback if < 2 results
FALLBACK_MIN_CONFIDENCE = 0.3  # Trigger fallback if max score < 0.3

# ==============================
# TIER 2: Semantic Query Caching
# ==============================

ENABLE_SEMANTIC_CACHE = True  # Cache embeddings & results
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.95  # Cosine similarity for cache hit
SEMANTIC_CACHE_TTL_HOURS = 24  # Cache validity period
SEMANTIC_CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "semantic_query_cache.db")

# ==============================
# TIER 2: Citation Tracking
# ==============================

ENABLE_CITATION_TRACKING = True  # Track which chunks support answer sentences
CITATION_IN_PROMPT = True  # Ask LLM to cite sources in answer

# ==============================
# TIER 3: Agentic Answer Refinement
# ==============================

ENABLE_TIER3_AGENTIC_RAG = True
TIER3_MAX_REFINE_ROUNDS = 2
TIER3_MIN_FAITHFULNESS = 0.7
TIER3_MIN_USEFULNESS = 0.65
TIER3_LOW_CONFIDENCE_TRIGGER = 0.55
TIER3_REFINE_TEMPERATURE = 0.1

# Languages for PaddleOCR
# 'en' works with multilingual by default, 
# but we’ll use multilingual model
OCR_LANG = "en"  # PaddleOCR will use multilingual detection

USE_GPU_FOR_OCR = False  # Change to True if GPU available
ENABLE_OCR = True

# If a PDF page has fewer extracted characters than this threshold,
# OCR fallback is triggered for that page.
PDF_OCR_FALLBACK_MIN_CHARS = 50

# ==============================
# LANGUAGE DETECTION
# ==============================

SUPPORTED_LANGUAGES = [
    "en",  # English
    "hi",  # Hindi
    "ta",  # Tamil
    "te",  # Telugu
    "kn",  # Kannada
    "bn",  # Bengali
    "ml",  # Malayalam
    "mr",  # Marathi
    "gu",  # Gujarati
    "pa",  # Punjabi
]

# ==============================
# PROMPT SETTINGS
# ==============================

SYSTEM_PROMPT_TEMPLATE = """
You are a multilingual document assistant.

Rules:
1. Answer ONLY from the provided context.
2. If answer is not in context, say you do not know.
3. Respond in exactly one final language (same as user language unless user explicitly requests another language).
4. Do not translate unless explicitly asked.
"""

# ==============================
# LLM JUDGE (Retrieval + Generation Audit)
# ==============================

ENABLE_LLM_JUDGE = True
LLM_JUDGE_MIN_OVERALL_SCORE = 0.6
LLM_JUDGE_TEMPERATURE = 0.0
LLM_JUDGE_MAX_RETRIES = 1

# Language compliance guardrail
ENABLE_LANGUAGE_COMPLIANCE_REWRITE = True
LANGUAGE_COMPLIANCE_MIN_CHARS = 40

# ==============================
# QUERY ROUTING + EVAL OBSERVABILITY
# ==============================

ENABLE_QUERY_INTENT_ROUTING = True
INTENT_ROUTER_DEFAULT_TOP_K = 5
INTENT_ROUTER_COMPLEX_TOP_K = 8
INTENT_ROUTER_SUMMARY_TOP_K = 10

ENABLE_GROUNDING_VERIFIER = True
GROUNDING_MIN_SUPPORT_RATIO = 0.55

ENABLE_EVAL_LOGGING = True
EVAL_LOG_PATH = os.path.join(PROCESSED_DATA_DIR, "rag_eval_log.jsonl")

# ==============================
# EVAL DASHBOARD
# ==============================

ENABLE_EVAL_DASHBOARD = True
EVAL_DASHBOARD_REFRESH_INTERVAL = 5  # Seconds between auto-refresh

# ==============================
# QUERY DECOMPOSITION (Multi-part Questions)
# ==============================

ENABLE_QUERY_DECOMPOSITION = True
QUERY_DECOMPOSER_MAX_SUB_QUERIES = 5
QUERY_DECOMPOSER_TEMPERATURE = 0.1  # Deterministic decomposition
QUERY_DECOMPOSER_MAX_TOKENS = 1000

# ==============================
# CITATION-GROUNDED GENERATION
# ==============================

ENABLE_CITATION_GROUNDING = True
ENABLE_CITATION_AUGMENTATION = True  # Add missing citations via LLM
CITATION_MIN_SUPPORT_RATIO = 0.7  # Acceptable claim support threshold
CITATION_AUGMENTATION_TEMPERATURE = 0.1

# ==============================
# HIERARCHICAL RETRIEVAL (Document-level First Pass)
# ==============================

ENABLE_HIERARCHICAL_RETRIEVAL = False  # Set True to prefer hierarchical over flat
HIERARCHICAL_TOP_DOCUMENTS = 5  # Stage 1: retrieve N documents
HIERARCHICAL_TOP_CHUNKS_PER_DOC = 3  # Stage 2: retrieve M chunks per document
HIERARCHICAL_DIVERSITY_PENALTY = 0.1  # Penalty for redundant chunks (0-1)
HIERARCHICAL_USE_DIVERSITY = True  # Apply diversity re-ranking

# ==============================
# STREAMLIT SETTINGS
# ==============================

APP_TITLE = "Indic Multilingual Multimodal RAG"

# Prompt budget controls for Groq request size
PROMPT_MAX_CONTEXT_CHARS = 4500
PROMPT_MAX_CONTEXT_CHUNKS = 5

# Default generation caps
RESPONSE_MAX_TOKENS = 900
STRUCTURED_RESPONSE_MAX_TOKENS = 1400

# ==============================
# GENERATION RELIABILITY
# ==============================

MCQ_RESPONSE_MAX_TOKENS = 2400
MCQ_CONTINUATION_MAX_ROUNDS = 3

# ==============================
# AUDIO / VIDEO (LOCAL ASR)
# ==============================

# Runs locally using faster-whisper (no external API needed)
ENABLE_AUDIO_VIDEO_INGESTION = True
ASR_MODEL_SIZE = "small"  # tiny, base, small, medium, large-v3
ASR_COMPUTE_TYPE = "int8"  # int8 / float16 / float32
ASR_DEVICE = "cpu"  # cpu / cuda
ASR_BEAM_SIZE = 1
ASR_VAD_FILTER = True
ASR_REDECODE_BEAM_SIZE = 2

# Segment-first AV processing (faster and more fault-tolerant than full-file decode)
ASR_SEGMENT_WINDOW_SECONDS = 30
ASR_SEGMENT_OVERLAP_SECONDS = 2

# Audio/video chunking strategy:
# - "segment_grouped": fast path, groups contiguous ASR segments into chunks
# - "full_text": current text chunker path (more normalization, slower)
# - "hybrid": segment groups first, then text chunker for very long groups
ASR_CHUNKING_STRATEGY = "segment_grouped"
ASR_SEGMENT_GROUP_SIZE = 5
ASR_MAX_CHARS_PER_CHUNK = 1200
ASR_PROGRESS_LOG_EVERY_SEGMENTS = 20

# Optional clip for faster testing on long media.
# Set 0 for full transcription.
ASR_MAX_TRANSCRIBE_SECONDS = 600
ASR_CACHE_ENABLED = True

# Quality fallback controls
ASR_ENABLE_QUALITY_FALLBACK = True
ASR_MIN_TEXT_QUALITY_SCORE = 0.48
ASR_FALLBACK_LANGUAGES = ["hi", "en", "ur", "bn", "ta", "te", "mr"]

# Optional forced language code (e.g. "hi"); set to None for auto-detect
ASR_FORCE_LANGUAGE = None

# Indic-centric language coverage used for guidance/debug labels
ASR_SUPPORTED_LANGUAGES = [
    "en", "hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "as", "ur"
]

# Optional benchmark behavior
ASR_BENCHMARK_ENABLE_WER = True