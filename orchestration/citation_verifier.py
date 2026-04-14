# citation_verifier.py
"""
Citation-Grounded Generation Module
Ensures generated answers include proper citations to source documents.
Verifies that cited chunks actually support the claims.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
from llm.client_factory import LLMClient


class CitationVerifier:
    """
    Verifies that answer claims are grounded in retrieved documents with proper citations.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize verifier with optional LLM client for citation augmentation."""
        self.llm_client = llm_client

    def verify_citations(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify that answer includes proper citations and claims are grounded.
        
        Args:
            answer: Generated answer text
            retrieved_docs: List of retrieved document chunks (with metadata)
            
        Returns:
            Dict with keys:
            - citations_found: List of cited doc indices
            - uncited_claims: List of sentences lacking citations
            - claim_support_ratio: float, fraction of claims with valid citations
            - issues: List of citation issues found
            - is_valid: bool, whether citations meet acceptance threshold
        """
        citations = self._extract_citations(answer)
        sentences = self._split_into_sentences(answer)
        
        cited_sentences = set()
        citation_docs = {}
        invalid_citations = []
        
        # Map each citation to its source doc
        for cit in citations:
            doc_idx = self._parse_citation_index(cit)
            if doc_idx is not None and 0 <= doc_idx < len(retrieved_docs):
                citation_docs[cit] = retrieved_docs[doc_idx]
                cited_sentences.update(self._find_sentences_with_citation(answer, cit))
            else:
                invalid_citations.append(cit)
        
        # Identify uncited sentences
        uncited = []
        for i, sent in enumerate(sentences):
            # Skip short sentences and intro/transition phrases
            if len(sent.split()) < 5 or self._is_transition_phrase(sent):
                continue
            if not any(self._sentence_has_citation_context(answer, sent) for _ in [None]):
                uncited.append(sent[:100])  # Truncate for readability
        
        claim_support_ratio = (
            (len(sentences) - len(uncited)) / len(sentences)
            if len(sentences) > 0
            else 1.0
        )
        
        issues = []
        if invalid_citations:
            issues.append(f"Invalid citation indices: {invalid_citations}")
        if len(uncited) > len(sentences) * 0.3:  # More than 30% uncited
            issues.append(f"{len(uncited)} sentences lack citations")
        
        return {
            "citations_found": list(citations),
            "uncited_claims": uncited[:5],  # Top 5
            "claim_support_ratio": min(1.0, max(0.0, claim_support_ratio)),
            "issues": issues,
            "is_valid": claim_support_ratio >= 0.7 and len(invalid_citations) == 0,
        }

    def augment_answer_with_citations(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Add missing citations to answer claims if LLM client is available.
        
        Args:
            answer: Original answer without citations
            retrieved_docs: List of retrieved documents
            
        Returns:
            Tuple of (augmented_answer, metadata)
        """
        if not self.llm_client:
            return answer, {"augmented": False, "reason": "no_llm_client"}
        
        # Build context reference
        doc_refs = self._build_doc_references(retrieved_docs)
        
        augmentation_prompt = (
            f"Rewrite the following answer to include proper citations. "
            f"Add [Source {i}] references after claims that are supported by the provided documents.\n\n"
            f"Available sources:\n{doc_refs}\n\n"
            f"Original answer:\n{answer}\n\n"
            f"Rewrite with citations:"
        )
        
        try:
            augmented = self.llm_client.generate(
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an expert at adding precise citations to factual claims. "
                            "Add [Source N] references after sentences or claims supported by the provided documents.",
                    },
                    {
                        "role": "user",
                        "content": augmentation_prompt,
                    },
                ],
                max_tokens=len(answer) // 4 + 500,  # Slightly longer for citations
                temperature=0.1,  # Conservative
            )
            
            return augmented or answer, {"augmented": True, "method": "llm_augmentation"}
        except Exception as e:
            return answer, {"augmented": False, "reason": f"augmentation_error: {e}"}

    def validate_citation_chain(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Deep validation: check that each cited claim is actually supported by the citation source.
        
        Args:
            answer: Answer with citations
            retrieved_docs: Retrieved documents
            
        Returns:
            Dict with claim-to-source mapping and validation scores
        """
        citations = self._extract_citations(answer)
        validation = {
            "total_citations": len(citations),
            "valid_citations": 0,
            "unsupported_citations": [],
            "claim_doc_mapping": {},
        }
        
        for cit in citations:
            doc_idx = self._parse_citation_index(cit)
            if doc_idx is None or doc_idx >= len(retrieved_docs):
                validation["unsupported_citations"].append(cit)
                continue
            
            doc = retrieved_docs[doc_idx]
            doc_content = doc.get("page_content", "").lower()
            
            # Find sentences citing this source
            sentences_with_cit = self._find_sentences_with_citation(answer, cit)
            
            supported_count = 0
            for sent in sentences_with_cit:
                sent_tokens = set(sent.lower().split())
                doc_tokens = set(doc_content.split())
                overlap = len(sent_tokens & doc_tokens) / len(sent_tokens) if sent_tokens else 0
                
                if overlap > 0.3:  # 30% token overlap heuristic
                    supported_count += 1
                else:
                    validation["unsupported_citations"].append(f"{cit} vs {sent[:50]}")
            
            if supported_count > 0:
                validation["valid_citations"] += 1
            
            validation["claim_doc_mapping"][cit] = {
                "doc_source": doc.get("metadata", {}).get("source", "unknown"),
                "supported_count": supported_count,
                "total_claims": len(sentences_with_cit),
            }
        
        return validation

    def _extract_citations(self, text: str) -> set:
        """Extract citation references from text (e.g., [Source 1], [1], [Doc 0])."""
        patterns = [
            r"\[Source\s*(\d+)\]",
            r"\[(\d+)\]",
            r"\[Doc\s*(\d+)\]",
        ]
        
        citations = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citations.add(match.group(0))
        
        return citations

    def _parse_citation_index(self, citation: str) -> int | None:
        """Extract numeric index from citation string."""
        match = re.search(r"(\d+)", citation)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return None
        return None

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple heuristic: split on [.!?] followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_sentences_with_citation(self, text: str, citation: str) -> List[str]:
        """Find sentences in text that contain the citation."""
        sentences = self._split_into_sentences(text)
        cited_sents = []
        
        # Look for citation in context (within 100 chars before/after)
        for sent in sentences:
            if citation in sent:
                cited_sents.append(sent)
        
        return cited_sents

    def _sentence_has_citation_context(self, text: str, sentence: str) -> bool:
        """Check if sentence appears near any citation in original text."""
        # Find sentence in text and check 200 char radius for citations
        match = text.find(sentence)
        if match == -1:
            return False
        
        window = text[max(0, match - 200):min(len(text), match + len(sentence) + 200)]
        return bool(self._extract_citations(window))

    def _is_transition_phrase(self, sentence: str) -> bool:
        """Check if sentence is a transition phrase rather than a factual claim."""
        phrases = [
            "based on the search results",
            "according to the documents",
            "as mentioned",
            "in conclusion",
            "to summarize",
            "therefore",
            "in summary",
        ]
        lower = sentence.lower()
        return any(phrase in lower for phrase in phrases)

    def _build_doc_references(self, docs: List[Dict[str, Any]]) -> str:
        """Build readable reference list from retrieved documents."""
        refs = []
        for i, doc in enumerate(docs[:5]):  # First 5 docs
            source = doc.get("metadata", {}).get("source", "unknown")
            page = doc.get("metadata", {}).get("page", "?")
            content_preview = doc.get("page_content", "")[:150]
            refs.append(f"Source {i}: {source} (pg. {page})\n  {content_preview}...")
        
        return "\n".join(refs)
