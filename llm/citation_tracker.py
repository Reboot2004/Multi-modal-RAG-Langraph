# llm/citation_tracker.py
"""
Citation Tracking

Track which source chunks support each sentence in the generated answer.
Enables:
- Transparency: "This fact came from document X"
- Fact-checking: Verify answer against sources
- Source attribution: Proper citations
- Hallucination detection: Sentences without source support
"""

import re
from typing import List, Dict, Tuple, Optional
from pipeline_logger import get_logger

logger = get_logger("citation_tracker")


class CitationTracker:
    """Track and verify answer citations"""

    def __init__(self):
        self.citation_marker_regex = re.compile(r'\[(\d+)\]')

    def annotate_sources(self, retrieved_chunks: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Annotate retrieved chunks with numeric citations for use in prompts.
        
        Returns:
            (annotated_chunks, numbered_sources_text)
        """
        annotated = []
        sources_text_parts = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            annotated_chunk = chunk.copy()
            annotated_chunk["citation_number"] = idx
            annotated_chunk["text_with_citation"] = f"[{idx}] {chunk.get('text', '')}"
            annotated.append(annotated_chunk)

            # Build sources reference text
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "N/A")
            sources_text_parts.append(f"[{idx}] {source}:page{page}")

        sources_text = "\n".join(sources_text_parts)

        logger.debug(
            "Citation tracker | annotated=%d | sources_lines=%d",
            len(annotated),
            len(sources_text_parts),
        )

        return annotated, sources_text

    def build_prompt_with_citations(
        self,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        """
        Build a prompt that encourages the LLM to cite sources.
        
        Returns:
            System/user prompt text with citation instructions
        """
        annotated, sources_text = self.annotate_sources(retrieved_chunks)

        prompt = f"""
You MUST cite your sources using [number] format from the provided sources.

For EVERY fact or claim, immediately add the citation [#] where # is the source number.

Example: "The capital is Paris [1]. It's located in France [2]."

Query: {query}

SOURCES:
{sources_text}

Context:
"""

        context_parts = []
        for chunk in annotated:
            cite_num = chunk.get("citation_number")
            text = chunk.get("text", "")[:400]
            context_parts.append(f"[{cite_num}] {text}")

        context = "\n\n".join(context_parts)
        prompt += context

        return prompt

    def extract_citations_from_answer(
        self, answer: str
    ) -> Tuple[List[Tuple[int, str]], List[str]]:
        """
        Extract citations from generated answer.
        
        Returns:
            (citations_list, unsupported_text_list)
            - citations_list: [(citation_num, text_snippet), ...]
            - unsupported_text_list: [sentences without citations]
        """
        citations = []
        unsupported = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find all citations in sentence
            matches = self.citation_marker_regex.findall(sentence)

            if matches:
                for match in matches:
                    citations.append((int(match), sentence[:80]))
            else:
                unsupported.append(sentence)

        logger.info(
            "Citation extraction | total_sentences=%d | cited=%d | unsupported=%d",
            len(sentences),
            len(citations),
            len(unsupported),
        )

        return citations, unsupported

    def verify_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict],
    ) -> Dict:
        """
        Verify that citations are valid and well-supported.
        
        Returns:
            {
                'valid_citations': [...],
                'invalid_citations': [...],
                'unsupported_sentences': [...],
                'coverage_percentage': float,
                'hallucination_risk': 'low' | 'medium' | 'high'
            }
        """
        citations, unsupported = self.extract_citations_from_answer(answer)

        # Check if citation numbers are valid
        max_valid_num = len(retrieved_chunks)
        valid_citations = []
        invalid_citations = []

        for cite_num, text_snippet in citations:
            if 1 <= cite_num <= max_valid_num:
                valid_citations.append({
                    'number': cite_num,
                    'text_snippet': text_snippet,
                    'source': retrieved_chunks[cite_num - 1].get('metadata', {}).get('source', 'Unknown'),
                })
            else:
                invalid_citations.append({
                    'number': cite_num,
                    'text_snippet': text_snippet,
                    'reason': f'citation_>_available_sources_{max_valid_num}',
                })

        # Calculate coverage
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        total_sentences = len([s for s in sentences if s.strip()])
        coverage_percentage = (
            (total_sentences - len(unsupported)) / max(1, total_sentences)
        ) * 100

        # Determine hallucination risk
        if coverage_percentage >= 90:
            hallucination_risk = 'low'
        elif coverage_percentage >= 70:
            hallucination_risk = 'medium'
        else:
            hallucination_risk = 'high'

        result = {
            'valid_citations': valid_citations,
            'invalid_citations': invalid_citations,
            'unsupported_sentences': unsupported[:5],  # Show first 5
            'coverage_percentage': coverage_percentage,
            'hallucination_risk': hallucination_risk,
        }

        logger.info(
            "Citation verification | valid=%d | invalid=%d | coverage=%.1f%% | risk=%s",
            len(valid_citations),
            len(invalid_citations),
            coverage_percentage,
            hallucination_risk,
        )

        return result

    def format_answer_with_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        """
        Format answer with inline source links and footnotes.
        
        Returns:
            Formatted answer with citations + footnote section
        """
        # Verify citations first
        verification = self.verify_citations(answer, retrieved_chunks)

        # Add hallucination warning if needed
        output = answer
        if verification['hallucination_risk'] in ['medium', 'high']:
            output = (
                f"⚠️ **Partial Coverage**: {verification['coverage_percentage']:.0f}% of statements are cited.\n\n"
                + output
            )

        # Add citations section
        output += "\n\n---\n**Sources Cited:**\n"
        for citation in verification['valid_citations']:
            output += f"- [{citation['number']}] {citation['source']}\n"

        if verification['unsupported_sentences']:
            output += "\n⚠️ **Unsupported claims** (consider re-querying):\n"
            for unsupported in verification['unsupported_sentences']:
                output += f"- {unsupported[:100]}\n"

        return output
