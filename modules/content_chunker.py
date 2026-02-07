# content_chunker.py - Document Chunking Module
# Adapted from LayIE-LLM chunking.py for optimal token management

import logging
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHUNK_SIZES, DEFAULT_CHUNK_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_index: int
    start_position: int  # Character position in original text
    end_position: int
    token_count: int
    element_paths: List[str] = field(default_factory=list)  # Paths of elements in this chunk
    
    def __repr__(self):
        return f"Chunk({self.chunk_index}, tokens={self.token_count}, chars={len(self.text)})"


class ContentChunker:
    """
    Chunks document content for optimal LLM processing.
    Based on LayIE-LLM paper findings (Table 3):
    - Medium-Max chunks (2048-4096 tokens) perform best
    - Zero-shot prompting achieves best F1
    """
    
    def __init__(self, chunk_size: str = DEFAULT_CHUNK_SIZE):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Size category - 'small', 'medium', or 'max'
        """
        self.chunk_size_category = chunk_size
        self.max_tokens = CHUNK_SIZES.get(chunk_size, CHUNK_SIZES['max'])
        
        # Reserve tokens for prompt template and response
        self.prompt_overhead = 1500  # Estimated tokens for prompt template
        self.response_reserve = 2000  # Reserve for response
        self.available_tokens = self.max_tokens - self.prompt_overhead
        
        # Initialize tokenizer (using cl100k_base for GPT-4)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoder: {e}. Using estimation.")
            self.tokenizer = None
        
        logger.info(f"Initialized chunker with {chunk_size} size ({self.max_tokens} tokens)")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: estimate ~1.3 tokens per word
            return int(len(text.split()) * 1.3)
    
    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[Chunk]:
        """
        Split text into optimal chunks for LLM processing.
        
        Args:
            text: Full document text to chunk
            preserve_sentences: If True, try to break at sentence boundaries
            
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []
        
        total_tokens = self.count_tokens(text)
        logger.info(f"Total text tokens: {total_tokens}, available per chunk: {self.available_tokens}")
        
        # If text fits in one chunk, return as single chunk
        if total_tokens <= self.available_tokens:
            return [Chunk(
                text=text,
                chunk_index=0,
                start_position=0,
                end_position=len(text),
                token_count=total_tokens
            )]
        
        # Split into chunks
        if preserve_sentences:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_tokens(text)
    
    def _chunk_by_sentences(self, text: str) -> List[Chunk]:
        """
        Chunk text by sentences, respecting token limits.
        """
        import re
        
        # Split by sentence-ending punctuation followed by space or newline
        sentence_pattern = r'(?<=[.!?])\s+|\n\n+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_start = 0
        current_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk limit, split it further
            if sentence_tokens > self.available_tokens:
                # Flush current chunk first
                if current_chunk_text:
                    chunks.append(Chunk(
                        text=current_chunk_text.strip(),
                        chunk_index=len(chunks),
                        start_position=chunk_start,
                        end_position=current_position,
                        token_count=current_chunk_tokens
                    ))
                    current_chunk_text = ""
                    current_chunk_tokens = 0
                    chunk_start = current_position
                
                # Split long sentence by tokens
                sub_chunks = self._split_long_text(sentence, current_position)
                for sc in sub_chunks:
                    sc.chunk_index = len(chunks)
                    chunks.append(sc)
                
                current_position += len(sentence) + 1
                chunk_start = current_position
                continue
            
            # Check if adding this sentence would exceed limit
            if current_chunk_tokens + sentence_tokens > self.available_tokens:
                # Save current chunk
                if current_chunk_text:
                    chunks.append(Chunk(
                        text=current_chunk_text.strip(),
                        chunk_index=len(chunks),
                        start_position=chunk_start,
                        end_position=current_position,
                        token_count=current_chunk_tokens
                    ))
                
                # Start new chunk with this sentence
                current_chunk_text = sentence + " "
                current_chunk_tokens = sentence_tokens
                chunk_start = current_position
            else:
                # Add to current chunk
                current_chunk_text += sentence + " "
                current_chunk_tokens += sentence_tokens
            
            current_position += len(sentence) + 1
        
        # Don't forget the last chunk
        if current_chunk_text.strip():
            chunks.append(Chunk(
                text=current_chunk_text.strip(),
                chunk_index=len(chunks),
                start_position=chunk_start,
                end_position=current_position,
                token_count=current_chunk_tokens
            ))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _chunk_by_tokens(self, text: str) -> List[Chunk]:
        """
        Chunk text by token count, breaking at word boundaries.
        """
        words = text.split()
        chunks = []
        current_words = []
        current_tokens = 0
        chunk_start = 0
        current_position = 0
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.available_tokens:
                # Save current chunk
                chunk_text = ' '.join(current_words)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    start_position=chunk_start,
                    end_position=current_position,
                    token_count=current_tokens
                ))
                
                # Start new chunk
                current_words = [word]
                current_tokens = word_tokens
                chunk_start = current_position
            else:
                current_words.append(word)
                current_tokens += word_tokens
            
            current_position += len(word) + 1
        
        # Last chunk
        if current_words:
            chunk_text = ' '.join(current_words)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                start_position=chunk_start,
                end_position=current_position,
                token_count=current_tokens
            ))
        
        return chunks
    
    def _split_long_text(self, text: str, start_position: int) -> List[Chunk]:
        """
        Split a long piece of text that exceeds chunk limit.
        """
        words = text.split()
        chunks = []
        current_words = []
        current_tokens = 0
        pos = start_position
        chunk_start = pos
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.available_tokens:
                chunk_text = ' '.join(current_words)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=0,  # Will be updated by caller
                    start_position=chunk_start,
                    end_position=pos,
                    token_count=current_tokens
                ))
                current_words = [word]
                current_tokens = word_tokens
                chunk_start = pos
            else:
                current_words.append(word)
                current_tokens += word_tokens
            
            pos += len(word) + 1
        
        if current_words:
            chunk_text = ' '.join(current_words)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=0,
                start_position=chunk_start,
                end_position=pos,
                token_count=current_tokens
            ))
        
        return chunks
    
    def chunk_with_overlap(self, text: str, overlap_tokens: int = 100) -> List[Chunk]:
        """
        Chunk text with overlapping sections for better context.
        Based on LayIE-LLM overlapping_chunks function.
        
        Args:
            text: Text to chunk
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List of Chunk objects with overlapping content
        """
        # First, create non-overlapping chunks
        base_chunks = self.chunk_text(text, preserve_sentences=True)
        
        if len(base_chunks) <= 1:
            return base_chunks
        
        # Add overlap from previous chunk to each chunk
        overlapped_chunks = []
        
        for i, chunk in enumerate(base_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Get end of previous chunk for overlap
            prev_chunk = base_chunks[i - 1]
            prev_words = prev_chunk.text.split()
            
            # Calculate words for overlap
            overlap_word_count = max(1, int(overlap_tokens / 1.3))
            overlap_words = prev_words[-overlap_word_count:] if len(prev_words) > overlap_word_count else prev_words
            overlap_text = ' '.join(overlap_words)
            
            # Create new chunk with overlap
            new_text = overlap_text + ' ' + chunk.text
            new_chunk = Chunk(
                text=new_text,
                chunk_index=i,
                start_position=chunk.start_position - len(overlap_text) - 1,
                end_position=chunk.end_position,
                token_count=self.count_tokens(new_text)
            )
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks


def chunk_document(text: str, chunk_size: str = 'max', use_overlap: bool = False) -> List[Chunk]:
    """
    Convenience function to chunk document text.
    
    Args:
        text: Document text to chunk
        chunk_size: Size category - 'small', 'medium', or 'max'
        use_overlap: Whether to use overlapping chunks
        
    Returns:
        List of Chunk objects
    """
    chunker = ContentChunker(chunk_size=chunk_size)
    
    if use_overlap:
        return chunker.chunk_with_overlap(text)
    else:
        return chunker.chunk_text(text)
