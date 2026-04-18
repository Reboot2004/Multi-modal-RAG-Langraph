import re
from typing import Callable, Dict, List, Set, TypedDict

from langgraph.graph import END, StateGraph

from config.settings import (
    LANGGRAPH_ENABLE_QUERY_EXPANSION,
    LANGGRAPH_EXPANSION_VARIANTS,
    LANGGRAPH_FINAL_TOP_K,
    LANGGRAPH_FUSION_RRF_K,
    LANGGRAPH_INITIAL_TOP_K,
    LANGGRAPH_MAX_CHUNKS_PER_SOURCE,
)
from llm.groq_client import GroqClient
from orchestration.adaptive_retrieval import AdaptiveRetrievalStrategy
from orchestration.query_decomposer import QueryDecomposer
from orchestration.query_contextualizer import QueryContextualizer
from orchestration.self_rag_gates import SelfRAGGates
from pipeline_logger import get_logger
from processing.language_detector import LanguageDetector
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever


logger = get_logger("langgraph_query")


class QueryGraphState(TypedDict, total=False):
    query: str
    original_query: str  # Track if query was contextualized
    query_contextualized: bool
    query_language: str
    response_language: str
    response_language_name: str
    response_script: str
    response_language_instruction: str
    response_language_reason: str
    query_variants: List[str]
    per_query_results: Dict[str, List[Dict]]
    fused_results: List[Dict]
    diversified_results: List[Dict]
    final_results: List[Dict]
    # Self-RAG confidence tracking
    retrieval_needed: bool
    doc_relevance_score: float
    faithfulness_score: float
    usefulness_score: float
    overall_confidence: float
    confidence_badge: str
    retrieval_top_k: int
    retrieval_mode: str


class LangGraphQueryOrchestrator:
    STOP_TERMS = {
        "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "is", "are",
        "was", "were", "be", "by", "from", "as", "what", "which", "who", "when", "where",
        "why", "how", "can", "could", "should", "would", "do", "does", "did", "it", "this",
        "that", "these", "those", "about", "into", "over", "under", "than", "then", "also",
    }

    STEP_ORDER = [
        "detect_language",
        "expand_query",
        "retrieve_candidates",
        "fuse_and_diversify",
        "rerank_final",
    ]

    def __init__(self, llm_client=None):
        self.lang_detector = LanguageDetector()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm_client = llm_client or GroqClient()
        self.self_rag_gates = SelfRAGGates(llm_client=self.llm_client)
        self.query_contextualizer = QueryContextualizer(llm_client=self.llm_client)
        self.query_decomposer = QueryDecomposer(llm_client=self.llm_client)
        self.adaptive_retrieval = AdaptiveRetrievalStrategy()

        self.enable_query_expansion = LANGGRAPH_ENABLE_QUERY_EXPANSION
        self.expansion_variants = max(1, int(LANGGRAPH_EXPANSION_VARIANTS))
        self.initial_top_k = max(5, int(LANGGRAPH_INITIAL_TOP_K))
        self.fusion_rrf_k = max(1, int(LANGGRAPH_FUSION_RRF_K))
        self.max_chunks_per_source = max(1, int(LANGGRAPH_MAX_CHUNKS_PER_SOURCE))
        self.final_top_k = max(1, int(LANGGRAPH_FINAL_TOP_K))

        self.graph = self._build_graph()
        logger.info(
            "LangGraphQueryOrchestrator initialized | expansion=%s | variants=%d | initial_top_k=%d | final_top_k=%d",
            self.enable_query_expansion,
            self.expansion_variants,
            self.initial_top_k,
            self.final_top_k,
        )
        self.progress_callback = None

    def _emit_progress(self, step_name: str):
        if self.progress_callback is None:
            return

        try:
            position = self.STEP_ORDER.index(step_name) + 1
            total = len(self.STEP_ORDER)
            self.progress_callback(step_name, position, total)
        except Exception:
            logger.debug("Progress callback failed for step=%s", step_name)

    def _build_graph(self):
        graph = StateGraph(QueryGraphState)

        graph.add_node("detect_language", self._detect_language)
        graph.add_node("contextualize_query", self._contextualize_query)
        graph.add_node("expand_query", self._expand_query)
        graph.add_node("retrieve_candidates", self._retrieve_candidates)
        graph.add_node("fuse_and_diversify", self._fuse_and_diversify)
        graph.add_node("rerank_final", self._rerank_final)
        graph.add_node("apply_self_rag_gates", self._apply_self_rag_gates)

        graph.set_entry_point("detect_language")
        graph.add_edge("detect_language", "contextualize_query")
        graph.add_edge("contextualize_query", "expand_query")
        graph.add_edge("expand_query", "retrieve_candidates")
        graph.add_edge("retrieve_candidates", "fuse_and_diversify")
        graph.add_edge("fuse_and_diversify", "rerank_final")
        graph.add_edge("rerank_final", "apply_self_rag_gates")
        graph.add_edge("apply_self_rag_gates", END)

        return graph.compile()

    def retrieve(self, query: str, top_k: int = None, progress_callback: Callable[[str, int, int], None] = None, conversation_history: List[Dict] = None) -> Dict:
        requested_top_k = self.final_top_k if top_k is None else max(1, int(top_k))
        self.progress_callback = progress_callback

        try:
            input_state = {"query": query}
            if conversation_history:
                input_state["conversation_history"] = conversation_history
            
            final_state = self.graph.invoke(input_state)
            results = final_state.get("final_results", [])[:requested_top_k]
            query_language = final_state.get("query_language", "en")
            response_language = final_state.get("response_language", query_language)

            if not results:
                logger.warning("LangGraph retrieval returned no results; using fallback retriever")
                fallback = self.retriever.retrieve(query, top_k=requested_top_k)
                response_resolution = self.lang_detector.resolve_response_language(
                    query,
                    fallback.get("query_language", "en"),
                )
                return {
                    "query_language": fallback.get("query_language", "en"),
                    "response_language": response_resolution.get("language_code", "en"),
                    "response_language_name": response_resolution.get("language_name", "English"),
                    "response_script": response_resolution.get("script", "Latin"),
                    "response_language_instruction": response_resolution.get("instruction", ""),
                    "response_language_reason": response_resolution.get("reason", "detected"),
                    "results": fallback.get("results", []),
                }

            return {
                "query_language": query_language,
                "response_language": response_language,
                "response_language_name": final_state.get("response_language_name", "English"),
                "response_script": final_state.get("response_script", "Latin"),
                "response_language_instruction": final_state.get("response_language_instruction", ""),
                "response_language_reason": final_state.get("response_language_reason", "detected"),
                "results": results,
                # Contextual retrieval info
                "original_query": final_state.get("original_query"),
                "query_contextualized": final_state.get("query_contextualized", False),
                "final_query": query,
                # Self-RAG confidence scores
                "doc_relevance_score": final_state.get("doc_relevance_score", 1.0),
                "faithfulness_score": final_state.get("faithfulness_score", 0.5),
                "usefulness_score": final_state.get("usefulness_score", 0.5),
                "overall_confidence": final_state.get("overall_confidence", 0.5),
                "confidence_badge": final_state.get("confidence_badge", "🟡"),
            }

        except Exception as ex:
            logger.exception("LangGraph retrieval failed; fallback to base retriever | error=%s", ex)
            fallback = self.retriever.retrieve(query, top_k=requested_top_k)
            response_resolution = self.lang_detector.resolve_response_language(
                query,
                fallback.get("query_language", "en"),
            )
            return {
                "query_language": fallback.get("query_language", "en"),
                "response_language": response_resolution.get("language_code", "en"),
                "response_language_name": response_resolution.get("language_name", "English"),
                "response_script": response_resolution.get("script", "Latin"),
                "response_language_instruction": response_resolution.get("instruction", ""),
                "response_language_reason": response_resolution.get("reason", "detected"),
                "results": fallback.get("results", []),
                # Fallback confidence scores
                "doc_relevance_score": 0.3,
                "faithfulness_score": 0.3,
                "usefulness_score": 0.3,
                "overall_confidence": 0.3,
                "confidence_badge": "🔴",
            }
        finally:
            self.progress_callback = None

    def _detect_language(self, state: QueryGraphState) -> QueryGraphState:
        self._emit_progress("detect_language")
        query = state.get("query", "")
        query_language = self.lang_detector.detect_language(query)
        response_resolution = self.lang_detector.resolve_response_language(query, query_language)
        logger.info(
            "LangGraph node detect_language | detected=%s | response=%s | reason=%s",
            query_language,
            response_resolution.get("language_code", "en"),
            response_resolution.get("reason", "detected"),
        )
        return {
            "query_language": query_language,
            "response_language": response_resolution.get("language_code", "en"),
            "response_language_name": response_resolution.get("language_name", "English"),
            "response_script": response_resolution.get("script", "Latin"),
            "response_language_instruction": response_resolution.get("instruction", ""),
            "response_language_reason": response_resolution.get("reason", "detected"),
        }

    def _contextualize_query(self, state: QueryGraphState) -> QueryGraphState:
        """Apply Contextual Retrieval: rewrite query using conversation history."""
        self._emit_progress("contextualize_query")
        query = state.get("query", "").strip()
        
        # Try to get conversation history if available
        # (This would be passed from app.py via session state)
        conversation_history = state.get("conversation_history", [])
        
        rewritten_query, was_rewritten, reason = self.query_contextualizer.contextualize_query(
            query,
            conversation_history if conversation_history else None,
        )
        
        if was_rewritten:
            logger.info(
                "LangGraph node contextualize_query | rewritten=true | reason=%s",
                reason,
            )
        else:
            logger.debug(
                "LangGraph node contextualize_query | rewritten=false | reason=%s",
                reason,
            )
        
        return {
            "query": rewritten_query,
            "original_query": query if was_rewritten else None,
            "query_contextualized": was_rewritten,
        }

    def _expand_query(self, state: QueryGraphState) -> QueryGraphState:
        self._emit_progress("expand_query")
        query = state.get("query", "").strip()
        query_language = state.get("query_language", "en")

        variants = [query] if query else []

        if query:
            decomposition = self.query_decomposer.decompose(query, max_sub_queries=4)
            sub_queries = decomposition.get("sub_queries", []) if decomposition.get("is_multi_part") else []
            if sub_queries:
                variants.extend(sub_queries)
                logger.info(
                    "LangGraph node expand_query | decomposition_applied=true | sub_queries=%d",
                    len(sub_queries),
                )

        corpus_size = int(getattr(self.retriever.vector_store.index, "ntotal", 0))
        query_word_count = len(query.split())

        if not query or corpus_size <= 3:
            logger.info(
                "LangGraph node expand_query | skipped=true | corpus_size=%d | words=%d",
                corpus_size,
                query_word_count,
            )
            return {"query_variants": self._dedupe_variants(variants)}

        should_generate_rewrites = (
            self.enable_query_expansion
            and self.expansion_variants > 1
            and query_word_count > 8
        )

        if not should_generate_rewrites:
            deduped = self._dedupe_variants(variants)
            logger.info("LangGraph node expand_query | variants=%d", len(deduped))
            logger.debug("Query variants: %s", deduped)
            return {"query_variants": deduped}

        additional_needed = max(0, self.expansion_variants - 1)
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite search queries for retrieval quality. "
                    "Return one rewritten query per line with no bullets or numbering."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n"
                    f"Language code: {query_language}\n"
                    f"Generate exactly {additional_needed} semantically diverse rewrites that preserve intent."
                ),
            },
        ]

        try:
            response = self.llm_client.generate(messages, max_tokens=220, temperature=0.2)
            for line in (response or "").splitlines():
                cleaned = line.strip().lstrip("-•0123456789. ").strip()
                if cleaned:
                    variants.append(cleaned)
        except Exception as ex:
            logger.warning("Query expansion failed; using original query only | error=%s", ex)

        deduped = self._dedupe_variants(variants)

        if not deduped:
            deduped = [query]

        deduped = deduped[: self.expansion_variants]
        logger.info("LangGraph node expand_query | variants=%d", len(deduped))
        logger.debug("Query variants: %s", deduped)
        return {"query_variants": deduped}

    def _retrieve_candidates(self, state: QueryGraphState) -> QueryGraphState:
        self._emit_progress("retrieve_candidates")
        query = state.get("query", "")
        query_variants = state.get("query_variants", [])
        adaptive_top_k, adaptive_reason = self.adaptive_retrieval.adaptive_top_k(
            query,
            base_k=self.initial_top_k,
        )
        adaptive_top_k = max(self.final_top_k * 2, adaptive_top_k)

        per_query_results = {}

        for variant in query_variants:
            retrieval = self.retriever.retrieve(variant, top_k=adaptive_top_k)
            per_query_results[variant] = retrieval.get("results", [])

        should_expand_depth = self._needs_deeper_retrieval(query_variants, per_query_results)
        retrieval_mode = adaptive_reason

        if should_expand_depth:
            expanded_top_k = min(max(adaptive_top_k + 10, self.initial_top_k + 8), max(40, self.initial_top_k * 2))
            retrieval_mode = f"{adaptive_reason}_evidence_expanded"
            for variant in query_variants:
                expanded = self.retriever.retrieve(variant, top_k=expanded_top_k).get("results", [])
                per_query_results[variant] = self._merge_by_chunk_id(
                    per_query_results.get(variant, []),
                    expanded,
                )
            adaptive_top_k = expanded_top_k

        total_candidates = sum(len(v) for v in per_query_results.values())
        logger.info(
            "LangGraph node retrieve_candidates | variants=%d | total_candidates=%d | top_k=%d | mode=%s",
            len(query_variants),
            total_candidates,
            adaptive_top_k,
            retrieval_mode,
        )
        return {
            "per_query_results": per_query_results,
            "retrieval_top_k": adaptive_top_k,
            "retrieval_mode": retrieval_mode,
        }

    def _fuse_and_diversify(self, state: QueryGraphState) -> QueryGraphState:
        self._emit_progress("fuse_and_diversify")
        query = state.get("query", "")
        per_query_results = state.get("per_query_results", {})

        fused_by_chunk = {}

        for _, results in per_query_results.items():
            for rank, item in enumerate(results, start=1):
                metadata = item.get("metadata", {})
                chunk_id = metadata.get("chunk_id")

                if chunk_id is None:
                    source = metadata.get("source", "unknown")
                    page = metadata.get("page", "unknown")
                    text_head = (item.get("text", "") or "")[:80]
                    chunk_id = f"{source}:{page}:{text_head}"

                if chunk_id not in fused_by_chunk:
                    fused_by_chunk[chunk_id] = {
                        "score": 0.0,
                        "text": item.get("text", ""),
                        "metadata": metadata,
                        "semantic_score": float(item.get("semantic_score", 0.0)),
                        "hype_score": float(item.get("hype_score", 0.0)),
                    }

                base_score = float(item.get("score", 0.0))
                rrf_score = 1.0 / (self.fusion_rrf_k + rank)
                fused_by_chunk[chunk_id]["score"] += base_score + rrf_score
                fused_by_chunk[chunk_id]["semantic_score"] = max(
                    fused_by_chunk[chunk_id]["semantic_score"],
                    float(item.get("semantic_score", 0.0)),
                )
                fused_by_chunk[chunk_id]["hype_score"] = max(
                    fused_by_chunk[chunk_id]["hype_score"],
                    float(item.get("hype_score", 0.0)),
                )

        fused_results = sorted(
            fused_by_chunk.values(),
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

        parent_counts = {}
        source_counts = {}
        diversified_results = []
        max_per_source = max(1, self.max_chunks_per_source - 1)
        target_pool = max(self.initial_top_k, self.final_top_k * 4)

        for item in fused_results:
            metadata = item.get("metadata", {})
            source = metadata.get("source", "unknown")
            parent_id = metadata.get("parent_id") or f"source::{source}"
            used = parent_counts.get(parent_id, 0)
            if used >= self.max_chunks_per_source:
                continue
            if source_counts.get(source, 0) >= max_per_source:
                continue

            parent_counts[parent_id] = used + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            diversified_results.append(item)

            if len(diversified_results) >= target_pool:
                break

        diversified_results = self._expand_parent_and_sibling_context(
            query,
            diversified_results,
            fused_results,
            target_pool=target_pool,
        )

        logger.info(
            "LangGraph node fuse_and_diversify | fused=%d | diversified=%d",
            len(fused_results),
            len(diversified_results),
        )

        return {
            "fused_results": fused_results,
            "diversified_results": diversified_results,
        }

    def _rerank_final(self, state: QueryGraphState) -> QueryGraphState:
        self._emit_progress("rerank_final")
        query = state.get("query", "")
        diversified = state.get("diversified_results", [])

        reranked = self.reranker.rerank(
            query,
            diversified,
            top_k=max(self.final_top_k * 4, self.final_top_k),
        )
        reranked = self._select_coverage_balanced_results(query, reranked, self.final_top_k)
        logger.info(
            "LangGraph node rerank_final | input=%d | output=%d",
            len(diversified),
            len(reranked),
        )
        return {"final_results": reranked}

    def _apply_self_rag_gates(self, state: QueryGraphState) -> QueryGraphState:
        """Apply Self-RAG quality gates and compute confidence scores."""
        query = state.get("query", "")
        final_results = state.get("final_results", [])

        # Initialize default scores
        scores = {
            "retrieval_needed": True,
            "doc_relevance_score": 1.0,
            "faithfulness_score": 0.5,
            "usefulness_score": 0.5,
            "overall_confidence": 0.5,
            "confidence_badge": "🟡",
        }

        # Gate 2: Check if retrieved docs are relevant
        if final_results:
            is_relevant, relevance_conf, reason = self.self_rag_gates.gate_doc_relevance(
                query, final_results
            )
            evidence_conf = self._estimate_evidence_completeness(query, final_results)
            scores["doc_relevance_score"] = max(0.0, min(1.0, (relevance_conf + evidence_conf) / 2.0))
            retrieval_conf = max(0.0, min(1.0, len(final_results) / max(1, self.final_top_k)))
            scores["faithfulness_score"] = max(0.45, evidence_conf * 0.9)
            scores["usefulness_score"] = max(0.45, evidence_conf)
            scores["overall_confidence"] = self.self_rag_gates.compute_overall_confidence(
                retrieval_confidence=retrieval_conf,
                doc_relevance_confidence=scores["doc_relevance_score"],
                faithfulness_confidence=scores["faithfulness_score"],
                usefulness_confidence=scores["usefulness_score"],
            )
            scores["confidence_badge"] = self.self_rag_gates.get_confidence_badge(
                scores["overall_confidence"]
            )[0]
            if not is_relevant:
                logger.warning(
                    "Self-RAG Gate 2 triggered low relevance | confidence=%.2f | reason=%s",
                    relevance_conf,
                    reason,
                )
        else:
            scores["doc_relevance_score"] = 0.0

        # For now, we'll apply faithfulness and usefulness gates after LLM generation
        # This method just computes doc relevance and sets defaults
        logger.info(
            "Self-RAG Gates applied | doc_relevance=%.2f | overall=%.2f | final_results=%d",
            scores["doc_relevance_score"],
            scores["overall_confidence"],
            len(final_results),
        )

        return scores

    def _dedupe_variants(self, variants: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for item in variants:
            key = (item or "").lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append((item or "").strip())
        return deduped

    def _needs_deeper_retrieval(self, query_variants: List[str], per_query_results: Dict[str, List[Dict]]) -> bool:
        all_results = []
        for items in per_query_results.values():
            all_results.extend(items)

        if not all_results:
            return True

        top_score = max(float(item.get("score", 0.0)) for item in all_results)
        unique_sources = {
            str(item.get("metadata", {}).get("source", "unknown"))
            for item in all_results[: max(8, self.final_top_k * 2)]
        }
        multi_hop_query = len(query_variants) > 1
        weak_signal = top_score < 0.28 or len(unique_sources) < min(2, len(all_results))
        return multi_hop_query or weak_signal

    def _merge_by_chunk_id(self, base_results: List[Dict], extra_results: List[Dict]) -> List[Dict]:
        merged = {}
        for item in base_results + extra_results:
            metadata = item.get("metadata", {})
            chunk_id = metadata.get("chunk_id")
            if chunk_id is None:
                source = metadata.get("source", "unknown")
                page = metadata.get("page", "unknown")
                chunk_id = f"{source}:{page}:{(item.get('text', '') or '')[:80]}"

            existing = merged.get(chunk_id)
            if existing is None or float(item.get("score", 0.0)) > float(existing.get("score", 0.0)):
                merged[chunk_id] = item

        return sorted(merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)

    def _expand_parent_and_sibling_context(
        self,
        query: str,
        diversified_results: List[Dict],
        fused_results: List[Dict],
        target_pool: int,
    ) -> List[Dict]:
        if len(diversified_results) >= target_pool:
            return diversified_results

        fused_by_chunk_id = {}
        for item in fused_results:
            chunk_id = item.get("metadata", {}).get("chunk_id")
            if chunk_id is not None and chunk_id not in fused_by_chunk_id:
                fused_by_chunk_id[chunk_id] = item

        selected_chunk_ids = {
            item.get("metadata", {}).get("chunk_id")
            for item in diversified_results
            if item.get("metadata", {}).get("chunk_id") is not None
        }
        selected_parent_ids = {
            item.get("metadata", {}).get("parent_id")
            for item in diversified_results
            if item.get("metadata", {}).get("parent_id")
        }

        query_terms = self._extract_query_terms(query)
        parent_candidates = []
        for meta in self.retriever.vector_store.metadata:
            parent_id = meta.get("parent_id")
            chunk_id = meta.get("chunk_id")
            if not parent_id or parent_id not in selected_parent_ids or chunk_id in selected_chunk_ids:
                continue
            parent_candidates.append(meta)

        scored_candidates = []
        for meta in parent_candidates:
            text = str(meta.get("text", ""))
            text_terms = self._extract_query_terms(text)
            overlap = len(query_terms.intersection(text_terms))
            overlap_bonus = min(0.25, overlap * 0.05)
            chunk_type = str(meta.get("chunk_type", "")).lower()
            summary_bonus = 0.08 if "summary" in chunk_type or "heading" in chunk_type else 0.0
            base_item = fused_by_chunk_id.get(meta.get("chunk_id"))
            base_score = float(base_item.get("score", 0.0)) if base_item else 0.0

            scored_candidates.append(
                (
                    base_score + overlap_bonus + summary_bonus,
                    {
                        "score": base_score + overlap_bonus + summary_bonus,
                        "semantic_score": float(base_item.get("semantic_score", 0.0)) if base_item else 0.0,
                        "hype_score": float(base_item.get("hype_score", 0.0)) if base_item else 0.0,
                        "lexical_score": float(base_item.get("lexical_score", 0.0)) if base_item else 0.0,
                        "text": text,
                        "metadata": meta,
                    },
                )
            )

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, item in scored_candidates:
            chunk_id = item.get("metadata", {}).get("chunk_id")
            if chunk_id in selected_chunk_ids:
                continue
            diversified_results.append(item)
            selected_chunk_ids.add(chunk_id)
            if len(diversified_results) >= target_pool:
                break

        return diversified_results

    def _extract_query_terms(self, text: str) -> Set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower()))
        return {t for t in tokens if t not in self.STOP_TERMS}

    def _select_coverage_balanced_results(self, query: str, reranked: List[Dict], top_k: int) -> List[Dict]:
        if len(reranked) <= top_k:
            return reranked[:top_k]

        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return reranked[:top_k]

        remaining = list(reranked)
        selected = []
        covered_terms = set()
        selected_sources = set()

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for idx, item in enumerate(remaining):
                item_terms = self._extract_query_terms(item.get("text", ""))
                new_terms = len((item_terms & query_terms) - covered_terms)
                source = str(item.get("metadata", {}).get("source", "unknown"))
                source_bonus = 0.05 if source not in selected_sources else 0.0
                rerank_score = float(item.get("rerank_score", item.get("score", 0.0)))
                composite = rerank_score + (new_terms * 0.04) + source_bonus

                if composite > best_score:
                    best_score = composite
                    best_idx = idx

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            selected_sources.add(str(chosen.get("metadata", {}).get("source", "unknown")))
            covered_terms.update(self._extract_query_terms(chosen.get("text", "")) & query_terms)

        return selected

    def _estimate_evidence_completeness(self, query: str, final_results: List[Dict]) -> float:
        if not final_results:
            return 0.0

        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return min(1.0, len(final_results) / max(1, self.final_top_k))

        covered = set()
        unique_sources = set()
        for item in final_results:
            covered.update(self._extract_query_terms(item.get("text", "")) & query_terms)
            unique_sources.add(str(item.get("metadata", {}).get("source", "unknown")))

        term_ratio = len(covered) / max(1, len(query_terms))
        source_ratio = min(1.0, len(unique_sources) / 2.0)
        return max(0.0, min(1.0, (term_ratio * 0.7) + (source_ratio * 0.3)))
