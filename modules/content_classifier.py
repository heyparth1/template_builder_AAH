# content_classifier.py - AI-Powered Content Classification Module
# Uses GPT to identify dynamic vs static content in documents

import logging
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    DYNAMIC_CONTENT_SCHEMA,
    STATIC_CONTENT_RULES,
    CLASSIFICATION_PROMPT_TEMPLATE
)
from modules.model_interaction import GPTClient, create_client
from modules.content_chunker import Chunk, chunk_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Replacement:
    """Represents a content replacement"""
    original: str
    placeholder: str
    category: str
    confidence: float = 1.0
    reason: str = ""
    chunk_index: int = -1
    position_in_chunk: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "placeholder": self.placeholder,
            "category": self.category,
            "confidence": self.confidence,
            "reason": self.reason
        }


@dataclass
class ClassificationResult:
    """Complete classification result for a document"""
    replacements: List[Replacement] = field(default_factory=list)
    chunks_processed: int = 0
    total_tokens_used: int = 0
    errors: List[str] = field(default_factory=list)
    
    def get_by_category(self, category: str) -> List[Replacement]:
        """Get all replacements of a specific category"""
        return [r for r in self.replacements if r.category == category]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics of replacements by category"""
        stats = {}
        for r in self.replacements:
            stats[r.category] = stats.get(r.category, 0) + 1
        return stats


class ContentClassifier:
    """
    Classifies document content to identify dynamic vs static content.
    Uses GPT-4 with zero-shot prompting (best F1 per LayIE-LLM paper).
    """
    
    def __init__(self, gpt_client: GPTClient = None):
        """
        Initialize the classifier.
        
        Args:
            gpt_client: Optional pre-configured GPT client
        """
        self.gpt_client = gpt_client or create_client()
        self.schema = DYNAMIC_CONTENT_SCHEMA
        self.static_rules = STATIC_CONTENT_RULES
    
    def _build_prompt(self, text_chunk: str) -> str:
        """
        Build classification prompt for a text chunk.
        Uses zero-shot approach (best F1 per paper Table 4).
        
        Args:
            text_chunk: Text to classify
            
        Returns:
            Complete prompt string
        """
        # Simplify schema for prompt
        schema_summary = {}
        for category, info in self.schema.items():
            schema_summary[category] = {
                "description": info["description"],
                "placeholder": info["placeholder"]
            }
        
        return CLASSIFICATION_PROMPT_TEMPLATE.format(
            text_chunk=text_chunk,
            schema_json=json.dumps(schema_summary, indent=2)
        )
    
    def _apply_regex_patterns(self, text: str) -> List[Replacement]:
        """
        Apply regex patterns for initial detection.
        Supplements LLM classification with pattern matching.
        
        Args:
            text: Text to search
            
        Returns:
            List of regex-detected replacements
        """
        replacements = []
        
        for category, info in self.schema.items():
            patterns = info.get("patterns", [])
            placeholder = info["placeholder"]
            
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        original = match.group()
                        # Check if this is likely dynamic (not a template placeholder)
                        if original and not original.startswith("{{"):
                            replacement = Replacement(
                                original=original,
                                placeholder=placeholder,
                                category=category,
                                confidence=0.8,  # Lower confidence for regex
                                reason="Detected by pattern matching"
                            )
                            replacements.append(replacement)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {category}: {e}")
        
        return replacements
    
    def _classify_chunk(self, chunk: Chunk, chunk_index: int) -> List[Replacement]:
        """
        Classify a single chunk using GPT.
        
        Args:
            chunk: Chunk to classify
            chunk_index: Index of this chunk
            
        Returns:
            List of identified replacements
        """
        prompt = self._build_prompt(chunk.text)
        
        try:
            result = self.gpt_client.classify_content(prompt)
            
            if "error" in result:
                logger.warning(f"Chunk {chunk_index} classification error: {result['error']}")
                # Fall back to regex
                return self._apply_regex_patterns(chunk.text)
            
            replacements = []
            for item in result.get("replacements", []):
                original = item.get("original", "")
                
                # Validate original text exists in chunk
                if original and original in chunk.text:
                    placeholder = item.get("placeholder", "{{UNKNOWN}}")
                    category = item.get("category", "UNKNOWN")
                    
                    # Normalize placeholder format
                    if not placeholder.startswith("{{"):
                        placeholder = f"{{{{{category}}}}}"
                    
                    replacement = Replacement(
                        original=original,
                        placeholder=placeholder,
                        category=category,
                        confidence=item.get("confidence", 0.9),
                        reason=item.get("reason", ""),
                        chunk_index=chunk_index,
                        position_in_chunk=chunk.text.find(original)
                    )
                    replacements.append(replacement)
                else:
                    logger.debug(f"Original text not found in chunk: {original[:50]}...")
            
            return replacements
            
        except Exception as e:
            logger.error(f"Error classifying chunk {chunk_index}: {e}")
            return self._apply_regex_patterns(chunk.text)
    
    def classify_document(
        self,
        text: str,
        chunk_size: str = 'max',
        use_parallel: bool = True,
        max_workers: int = 3,
        use_hybrid: bool = True  # NEW: Enable hybrid paragraph detection
    ) -> ClassificationResult:
        """
        Classify an entire document using hybrid approach.
        
        Hybrid approach:
        1. Pre-detect paragraphs with trigger phrases (fast, catches long paragraphs)
        2. Use LLM to classify chunks (finds values and confirms paragraphs)
        3. Merge results and deduplicate
        
        Args:
            text: Full document text
            chunk_size: Chunk size category
            use_parallel: Whether to process chunks in parallel
            max_workers: Number of parallel workers
            use_hybrid: Use hybrid paragraph detection (recommended)
            
        Returns:
            ClassificationResult with all replacements
        """
        logger.info(f"Starting document classification ({len(text)} chars)")
        
        result = ClassificationResult()
        all_replacements = []
        
        # === PHASE 1: Trigger-based paragraph detection ===
        if use_hybrid:
            try:
                from modules.paragraph_detector import detect_client_paragraphs
                paragraph_replacements = detect_client_paragraphs(text, min_confidence=0.5)
                
                for item in paragraph_replacements:
                    replacement = Replacement(
                        original=item["original"],
                        placeholder=item["placeholder"],
                        category=item["category"],
                        confidence=item["confidence"],
                        reason=item["reason"]
                    )
                    all_replacements.append(replacement)
                
                logger.info(f"Phase 1 (paragraphs): {len(paragraph_replacements)} detected by triggers")
            except ImportError:
                 # Module not found matches expected behavior if not installed
                 logger.debug("Paragraph detection module not found, skipping Phase 1.")
            except Exception as e:
                logger.warning(f"Paragraph detection failed: {e}")
        
        # === PHASE 2: Regex patterns for values ===
        regex_replacements = self._apply_regex_patterns(text)
        all_replacements.extend(regex_replacements)
        logger.info(f"Phase 2 (regex): {len(regex_replacements)} detected by patterns")
        
        # === PHASE 3: LLM classification for confirmation and additional detection ===
        chunks = chunk_document(text, chunk_size=chunk_size)
        logger.info(f"Phase 3 (LLM): Processing {len(chunks)} chunks")
        result.chunks_processed = len(chunks)
        
        if use_parallel and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._classify_chunk, chunk, i): i
                    for i, chunk in enumerate(chunks)
                }
                
                for future in as_completed(futures):
                    chunk_index = futures[future]
                    try:
                        replacements = future.result()
                        all_replacements.extend(replacements)
                        logger.info(f"Chunk {chunk_index}: {len(replacements)} replacements")
                    except Exception as e:
                        error_msg = f"Chunk {chunk_index} failed: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)
        else:
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
                replacements = self._classify_chunk(chunk, i)
                all_replacements.extend(replacements)
                logger.info(f"Chunk {i}: {len(replacements)} replacements")
        
        # === PHASE 4: Deduplicate and reconcile ===
        result.replacements = self._deduplicate_replacements(all_replacements)
        
        logger.info(f"Classification complete: {len(result.replacements)} unique replacements")
        logger.info(f"Statistics: {result.get_statistics()}")
        
        return result
    
    def _deduplicate_replacements(self, replacements: List[Replacement]) -> List[Replacement]:
        """
        Remove duplicate replacements and resolve conflicts.
        Based on LayIE-LLM reconcile_predictions approach.
        
        Args:
            replacements: List of all detected replacements
            
        Returns:
            Deduplicated list
        """
        if not replacements:
            return []
        
        # Group by original text
        by_original = {}
        for r in replacements:
            key = r.original.strip().lower()
            if key not in by_original:
                by_original[key] = []
            by_original[key].append(r)
        
        # For each group, pick the best replacement
        unique = []
        for key, group in by_original.items():
            if len(group) == 1:
                unique.append(group[0])
            else:
                # Pick the one with highest confidence
                best = max(group, key=lambda r: r.confidence)
                unique.append(best)
        
        # Sort by length (longer replacements first to avoid partial matches)
        unique.sort(key=lambda r: len(r.original), reverse=True)
        
        # Remove overlapping shorter matches
        final = []
        replaced_ranges = []  # Track what text has been claimed
        
        for r in unique:
            # Check if this text is contained within a longer match
            is_contained = any(
                r.original in other.original and r.original != other.original
                for other in final
            )
            
            if not is_contained:
                final.append(r)
        
        return final
    
    def classify_with_fallback(self, text: str) -> ClassificationResult:
        """
        Classify with fallback to regex if LLM fails.
        
        Args:
            text: Document text
            
        Returns:
            ClassificationResult
        """
        try:
            return self.classify_document(text)
        except Exception as e:
            logger.error(f"LLM classification failed, using regex fallback: {e}")
            
            result = ClassificationResult()
            result.replacements = self._apply_regex_patterns(text)
            result.errors.append(f"LLM failed: {e}")
            
            return result


def classify_content(text: str, chunk_size: str = 'max') -> ClassificationResult:
    """
    Convenience function to classify document content.
    
    Args:
        text: Document text to classify
        chunk_size: Chunk size category
        
    Returns:
        ClassificationResult
    """
    classifier = ContentClassifier()
    return classifier.classify_document(text, chunk_size=chunk_size)
