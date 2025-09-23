"""
Text Chunking Utilities
Advanced text processing and chunking for RAG systems
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ChunkConfig:
    """Configuration for text chunking behavior"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    preserve_sentence_boundaries: bool = True
    preserve_paragraph_boundaries: bool = True


class TextChunker:
    """
    Advanced text chunker with multiple strategies
    Supports fixed-size, semantic, and sentence-aware chunking
    """

    def __init__(self, config: ChunkConfig):
        """
        Initialize chunker with configuration

        Args:
            config: ChunkConfig instance with chunking parameters
        """
        self.config = config

        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        preserve_boundaries: bool = None
    ) -> List[str]:
        """
        Chunk text into smaller segments

        Args:
            text: Input text to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
            preserve_boundaries: Override sentence boundary preservation

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap
        preserve_boundaries = (
            preserve_boundaries
            if preserve_boundaries is not None
            else self.config.preserve_sentence_boundaries
        )

        if preserve_boundaries:
            return self._chunk_with_sentence_boundaries(text, chunk_size, chunk_overlap)
        else:
            return self._chunk_fixed_size(text, chunk_size, chunk_overlap)

    def _chunk_fixed_size(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Simple fixed-size chunking without boundary preservation

        Args:
            text: Input text
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Move start position considering overlap
            start = end - chunk_overlap

        return chunks

    def _chunk_with_sentence_boundaries(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """
        Chunk text while preserving sentence boundaries

        Args:
            text: Input text
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        sentence_buffer = []

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
                sentence_buffer.append(sentence)
            else:
                # Current chunk is ready
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    sentence_buffer, chunk_overlap
                )
                current_chunk = " ".join(overlap_sentences + [sentence])
                sentence_buffer = overlap_sentences + [sentence]

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk) >= self.config.min_chunk_size]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Handle paragraph breaks if configured
        if self.config.preserve_paragraph_boundaries:
            paragraphs = self.paragraph_breaks.split(text)
            sentences = []
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    para_sentences = self.sentence_endings.split(paragraph)
                    # Clean up and add sentences
                    for i, sentence in enumerate(para_sentences):
                        sentence = sentence.strip()
                        if sentence:
                            # Add back sentence ending except for last sentence
                            if i < len(para_sentences) - 1:
                                # Find the original ending
                                ending_match = self.sentence_endings.search(paragraph)
                                if ending_match:
                                    sentence += ending_match.group(0).strip()
                            sentences.append(sentence)
                    # Add paragraph break marker
                    if len(sentences) > 0:
                        sentences[-1] += "\n"
        else:
            sentences = self.sentence_endings.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """
        Get sentences for overlap based on character count

        Args:
            sentences: List of sentences
            overlap_size: Target overlap size in characters

        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []

        overlap_sentences = []
        current_overlap = 0

        # Work backwards from the end
        for sentence in reversed(sentences):
            if current_overlap + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                current_overlap += len(sentence)
            else:
                break

        return overlap_sentences

    def chunk_document(
        self,
        document: Dict[str, Any],
        preserve_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document while preserving metadata

        Args:
            document: Document with 'content' and optional metadata
            preserve_metadata: Whether to preserve original metadata

        Returns:
            List of chunk dictionaries with content and metadata
        """
        content = document.get("content", "")
        metadata = document.get("metadata", {}) if preserve_metadata else {}

        chunks = self.chunk_text(content)
        chunk_documents = []

        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "original_document_id": document.get("id"),
                }
            }
            chunk_documents.append(chunk_doc)

        return chunk_documents

    def merge_chunks(self, chunks: List[str], separator: str = " ") -> str:
        """
        Merge chunks back into a single text

        Args:
            chunks: List of text chunks
            separator: Separator between chunks

        Returns:
            Merged text
        """
        return separator.join(chunks)

    def get_chunk_stats(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Get statistics about a set of chunks

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }

        chunk_sizes = [len(chunk) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_sizes": chunk_sizes
        }