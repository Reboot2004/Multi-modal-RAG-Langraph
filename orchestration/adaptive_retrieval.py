# orchestration/adaptive_retrieval.py
"""
Adaptive Retrieval Strategy

Dynamically adjust retrieval parameters based on query characteristics:
- Query complexity detection
- Adaptive top-k selection
- Strategy switching (semantic vs hybrid vs lexical)
- Result threshold tuning
"""

from typing import Dict, Tuple
from pipeline_logger import get_logger

logger = get_logger("adaptive_retrieval")


class AdaptiveRetrievalStrategy:
    """Intelligent retrieval parameter adjustment"""

    def __init__(self):
        self.default_k = 5
        # Query complexity thresholds
        self.MIN_WORDS = 1
        self.SIMPLE_WORDS = 5  # Queries < 5 words are simple
        self.COMPLEX_WORDS = 15  # Queries > 15 words are complex

    def analyze_query_complexity(self, query: str) -> Dict:
        """
        Analyze query to determine complexity level.
        
        Returns:
            {
                'complexity': 'simple' | 'moderate' | 'complex',
                'word_count': int,
                'has_multipart': bool,  # "and", "or", "but", "how"
                'has_numerical': bool,  # "number", "count", "many"
                'has_comparative': bool,  # "vs", "between", "different"
                'confidence': float
            }
        """
        if not query:
            return {
                'complexity': 'simple',
                'word_count': 0,
                'has_multipart': False,
                'has_numerical': False,
                'has_comparative': False,
                'confidence': 0.5,
            }

        query_lower = query.lower()
        words = query_lower.split()
        word_count = len(words)

        # Analyze query characteristics
        multipart_keywords = {'and', 'or', 'but', 'how', 'why', 'explain', 'compare', 'relate'}
        numerical_keywords = {'number', 'count', 'many', 'much', 'percentage', 'ratio', 'average'}
        comparative_keywords = {'vs', 'versus', 'different', 'between', 'contrast', 'compare', 'like', 'unlike'}

        has_multipart = any(kw in query_lower for kw in multipart_keywords)
        has_numerical = any(kw in query_lower for kw in numerical_keywords)
        has_comparative = any(kw in query_lower for kw in comparative_keywords)

        # Determine complexity
        complexity_score = 0
        if word_count > self.COMPLEX_WORDS:
            complexity_score += 2
        elif word_count > self.SIMPLE_WORDS:
            complexity_score += 1

        if has_multipart:
            complexity_score += 1
        if has_numerical or has_comparative:
            complexity_score += 1

        if complexity_score >= 3:
            complexity = 'complex'
        elif complexity_score >= 1:
            complexity = 'moderate'
        else:
            complexity = 'simple'

        logger.debug(
            "Query complexity | text=%d chars | words=%d | complexity=%s | multipart=%s | numerical=%s",
            len(query),
            word_count,
            complexity,
            has_multipart,
            has_numerical,
        )

        return {
            'complexity': complexity,
            'word_count': word_count,
            'has_multipart': has_multipart,
            'has_numerical': has_numerical,
            'has_comparative': has_comparative,
            'confidence': min(0.95, 0.5 + (complexity_score * 0.15)),
        }

    def adaptive_top_k(self, query: str, base_k: int = None) -> Tuple[int, str]:
        """
        Determine optimal top_k based on query complexity.
        
        Returns:
            (adjusted_k, reason)
        """
        if base_k is None:
            base_k = self.default_k

        analysis = self.analyze_query_complexity(query)
        complexity = analysis['complexity']

        if complexity == 'simple':
            # Simple one-off queries, get fewer results (faster)
            adjusted_k = max(1, base_k - 2)
            reason = "simple_query"
        elif complexity == 'moderate':
            # Normal case
            adjusted_k = base_k
            reason = "moderate_query"
        else:  # complex
            # Complex multi-hop queries, get more candidates
            adjusted_k = min(10, base_k + 3)
            reason = "complex_query"

        logger.info(
            "Adaptive top_k | base=%d | adjusted=%d | reason=%s | complexity=%s",
            base_k,
            adjusted_k,
            reason,
            complexity,
        )

        return adjusted_k, reason

    def adaptive_strategy(self, query: str) -> Tuple[str, Dict]:
        """
        Recommend retrieval strategy based on query characteristics.
        
        Returns:
            (strategy, config)
            strategy: 'hybrid' | 'semantic' | 'lexical' | 'dense'
            config: strategy-specific parameters
        """
        analysis = self.analyze_query_complexity(query)
        complexity = analysis['complexity']

        if analysis['has_numerical'] or analysis['has_comparative']:
            # Numerical/comparative queries benefit from exact keyword matching
            strategy = 'hybrid'  # BM25 + semantic
            config = {
                'use_semantic': True,
                'use_lexical': True,
                'semantic_weight': 0.4,
                'lexical_weight': 0.6,  # Higher weight on exact matches
            }
            reason = "numerical_or_comparative"
        elif complexity == 'simple':
            # Simple queries: pure semantic is fast and accurate
            strategy = 'semantic'
            config = {
                'use_semantic': True,
                'use_lexical': False,
            }
            reason = "simple_semantic_only"
        elif complexity == 'complex':
            # Complex queries need all signals
            strategy = 'hybrid'
            config = {
                'use_semantic': True,
                'use_lexical': True,
                'use_hype': True,
                'semantic_weight': 0.5,
                'lexical_weight': 0.3,
                'hype_weight': 0.2,
            }
            reason = "complex_all_signals"
        else:
            # Moderate: balanced hybrid
            strategy = 'hybrid'
            config = {
                'use_semantic': True,
                'use_lexical': True,
                'semantic_weight': 0.6,
                'lexical_weight': 0.4,
            }
            reason = "moderate_balanced"

        logger.info(
            "Adaptive strategy | complexity=%s | strategy=%s | reason=%s",
            complexity,
            strategy,
            reason,
        )

        return strategy, config

    def adaptive_confidence_threshold(self, query: str) -> Tuple[float, str]:
        """
        Determine minimum confidence threshold for results.
        
        Returns:
            (threshold, reason)
        """
        analysis = self.analyze_query_complexity(query)
        complexity = analysis['complexity']

        if complexity == 'simple':
            # Simple queries: accept lower confidence (more results)
            threshold = 0.3
            reason = "simple_lower_threshold"
        elif complexity == 'moderate':
            # Moderate: balanced
            threshold = 0.5
            reason = "moderate_balanced"
        else:  # complex
            # Complex queries: require higher confidence
            threshold = 0.6
            reason = "complex_higher_threshold"

        logger.debug(
            "Adaptive confidence threshold | threshold=%.2f | reason=%s",
            threshold,
            reason,
        )

        return threshold, reason
