"""
Citation Tracking: Map answer sentences to source chunks.
Provides transparency and verifiability for RAG responses.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple


class CitationTracker:
    """Track and map answer sentences to source chunks."""

    def __init__(self, enable_citation_in_prompt: bool = True):
        """
        Initialize citation tracker.
        
        Args:
            enable_citation_in_prompt: Whether to prompt LLM for inline citations
        """
        self.enable_citation_in_prompt = enable_citation_in_prompt
        self.citations: Dict[int, List[str]] = {}  # sentence_idx -> chunk_ids
        self.chunk_content: Dict[str, str] = {}  # chunk_id -> chunk text

    def add_chunk(self, chunk_id: str, content: str, metadata: Dict[str, Any]):
        """Store chunk for later reference."""
        self.chunk_content[chunk_id] = {
            "content": content,
            "metadata": metadata,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", None),
        }

    def build_citation_prompt_suffix(self) -> str:
        """
        Build a prompt suffix asking LLM to include citations inline.
        
        Returns:
            Prompt instruction string
        """
        if not self.enable_citation_in_prompt:
            return ""

        return """

---
CITATION INSTRUCTIONS:
For each claim you make, include an inline citation in the format: [CITE source.pdf page 5] or [CITE document]
Example: "The capital of France is Paris [CITE travel_guide.pdf page 12]."
This helps users verify your sources.
---"""

    def extract_citations_from_answer(self, answer: str) -> List[Dict[str, Any]]:
        """
        Extract inline citations from answer text.
        
        Args:
            answer: Answer text potentially containing [CITE ...] tags
            
        Returns:
            List of extracted citations
        """
        citations = []
        
        # Pattern: [CITE source.pdf page N] or [CITE source]
        pattern = r"\[CITE\s+([^\]]+)\]"
        matches = re.finditer(pattern, answer)
        
        for match in matches:
            cite_text = match.group(1)
            citations.append({
                "raw_text": match.group(0),
                "source": cite_text,
                "span": match.span(),
            })
        
        return citations

    def map_answer_to_chunks(
        self,
        answer: str,
        source_chunks: List[Dict[str, Any]],
        llm_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Create comprehensive citation mapping.
        
        Args:
            answer: Generated answer text
            source_chunks: List of chunks used for answering
            llm_mapping: Optional LLM-generated mapping of sentences -> chunks
            
        Returns:
            Citation mapping with metadata
        """
        # Split answer into sentences
        sentences = self._split_into_sentences(answer)
        
        citation_map = {
            "answer": answer,
            "sentences": sentences,
            "sentence_citations": [],
            "chunk_sources": [],
            "extraction_method": "llm" if llm_mapping else "semantic",
        }
        
        # Store chunk info
        for chunk in source_chunks:
            chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
            if chunk_id:
                self.add_chunk(chunk_id, chunk.get("content", ""), chunk.get("metadata", {}))
                citation_map["chunk_sources"].append({
                    "id": chunk_id,
                    "source": chunk.get("metadata", {}).get("source", ""),
                    "page": chunk.get("metadata", {}).get("page", None),
                })
        
        # If LLM provided explicit mapping, use it
        if llm_mapping:
            citation_map["sentence_citations"] = llm_mapping
            return citation_map
        
        # Otherwise, use semantic similarity heuristic
        for sent_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Find most similar chunk
            best_chunk = self._find_best_chunk(sentence, source_chunks)
            if best_chunk:
                citation_map["sentence_citations"].append({
                    "sentence_idx": sent_idx,
                    "sentence": sentence,
                    "cited_chunks": [best_chunk["id"]],
                    "confidence": "low",  # Heuristic-based
                })
        
        return citation_map

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_best_chunk(
        self, sentence: str, chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find chunk most similar to sentence."""
        if not chunks:
            return None
        
        best_chunk = None
        best_score = 0
        
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue
            
            # Simple word overlap score
            sentence_words = set(sentence.lower().split())
            chunk_words = set(content.lower().split())
            
            if len(sentence_words) == 0:
                continue
            
            overlap = len(sentence_words & chunk_words)
            score = overlap / len(sentence_words)
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        return best_chunk

    def format_answer_with_citations(
        self, citation_map: Dict[str, Any], inline: bool = True
    ) -> str:
        """
        Format answer with visible citations.
        
        Args:
            citation_map: Pre-computed citation mapping
            inline: If True, add footnotes. If False, return as-is.
            
        Returns:
            Formatted answer string
        """
        if not inline:
            return citation_map["answer"]
        
        answer = citation_map["answer"]
        citations = citation_map["sentence_citations"]
        
        if not citations:
            return answer
        
        # Build footnote section
        footnotes = "\n\n**Sources:**\n"
        chunk_refs = {}  # chunk_id -> reference number
        ref_counter = 1
        
        for citation in citations:
            for chunk_id in citation.get("cited_chunks", []):
                if chunk_id not in chunk_refs:
                    chunk_refs[chunk_id] = ref_counter
                    ref_counter += 1
        
        for chunk_id, ref_num in chunk_refs.items():
            chunk_info = self.chunk_content.get(chunk_id, {})
            source = chunk_info.get("source", "Unknown")
            page = chunk_info.get("page")
            page_str = f" (page {page})" if page else ""
            footnotes += f"\n[{ref_num}] {source}{page_str}"
        
        return answer + footnotes

    def generate_span_citations(self, answer: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate span-level citations for interactive highlighting.
        
        Returns:
            (answer, list of citation spans with metadata)
        """
        citations_json = self.extract_citations_from_answer(answer)
        
        # Remove [CITE ...] from display answer
        display_answer = re.sub(r"\[CITE[^\]]*\]", "", answer).strip()
        
        spans = []
        for cite in citations_json:
            spans.append({
                "text": cite["source"],
                "position": cite["span"],
                "citing_context": display_answer[max(0, cite["span"][0]-50):cite["span"][1]+50],
            })
        
        return display_answer, spans
