"""
Smart Chunker - Structure-aware document chunking with semantic boundaries.

Improvements over basic chunking:
1. Semantic boundary detection - splits at natural topic shifts
2. Sliding window context - maintains overlap for continuity
3. Section-aware chunking - respects document structure
4. Table/figure preservation - keeps structured content intact
5. Adaptive chunk sizing - adjusts based on content density
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata for retrieval."""
    content: str
    chunk_id: str
    document_type: str  # 'criteria' or 'report'
    token_count: int
    
    # Location metadata
    page_start: int = 1
    page_end: int = 1
    section_title: Optional[str] = None
    section_number: Optional[str] = None  # e.g., "1", "2", "11"
    
    # Content metadata
    chunk_type: str = 'text'  # 'text', 'table', 'heading', 'mixed'
    has_figure_reference: bool = False
    figure_ids: List[str] = field(default_factory=list)
    
    # For criteria documents
    criterion_id: Optional[str] = None
    rubric_level: Optional[str] = None  # 'pass', 'merit', 'distinction'
    
    # NEW: Enhanced metadata for better retrieval
    keywords: List[str] = field(default_factory=list)  # Key terms in chunk
    parent_section: Optional[str] = None  # Parent heading for hierarchy
    chunk_index: int = 0  # Position in document
    
    # Extra metadata for filtering
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector store."""
        return {
            'content': self.content,
            'chunk_id': self.chunk_id,
            'document_type': self.document_type,
            'token_count': self.token_count,
            'page_start': self.page_start,
            'page_end': self.page_end,
            'section_title': self.section_title or '',
            'section_number': self.section_number or '',
            'chunk_type': self.chunk_type,
            'has_figure_reference': self.has_figure_reference,
            'criterion_id': self.criterion_id or '',
            'rubric_level': self.rubric_level or '',
            'keywords': ','.join(self.keywords) if self.keywords else '',
            'parent_section': self.parent_section or '',
            'chunk_index': self.chunk_index,
        }


class SmartChunker:
    """
    Smart document chunker with semantic boundary detection.
    
    Features:
    - Semantic boundary detection at paragraph/topic shifts
    - Section-aware chunking that respects document structure
    - Sliding window overlap for context continuity
    - Adaptive sizing based on content type
    - Keyword extraction for hybrid search
    """
    
    # Semantic boundary indicators
    BOUNDARY_PATTERNS = [
        r'^#{1,6}\s+',  # Markdown headings
        r'^\d+\.\s+[A-Z]',  # Numbered sections
        r'^(?:Introduction|Methodology|Results|Discussion|Conclusion|Summary|Background|Overview)',
        r'^(?:Section|Part|Chapter)\s+\d+',
        r'^\*\*[A-Z]',  # Bold headings in markdown
        r'^(?:First|Second|Third|Finally|In conclusion|To summarize)',
    ]
    
    # Topic shift indicators (when these appear, consider splitting)
    TOPIC_SHIFT_PATTERNS = [
        r'(?:However|Nevertheless|On the other hand|In contrast|Conversely)',
        r'(?:Furthermore|Moreover|Additionally|In addition)',
        r'(?:Therefore|Thus|Hence|Consequently|As a result)',
        r'(?:For example|For instance|Specifically|In particular)',
    ]
    
    def __init__(
        self,
        criteria_chunk_size: int = 400,
        criteria_overlap: int = 50,
        report_chunk_size: int = 600,
        report_overlap: int = 120,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker with configurable parameters.
        
        Args:
            criteria_chunk_size: Target tokens for criteria chunks
            criteria_overlap: Overlap tokens for criteria
            report_chunk_size: Target tokens for report chunks
            report_overlap: Overlap tokens for reports
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
            max_chunk_size: Maximum chunk size (hard limit)
            encoding_name: Tiktoken encoding
        """
        self.criteria_chunk_size = criteria_chunk_size
        self.criteria_overlap = criteria_overlap
        self.report_chunk_size = report_chunk_size
        self.report_overlap = report_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Compile patterns
        self.boundary_pattern = re.compile(
            '|'.join(self.BOUNDARY_PATTERNS), 
            re.MULTILINE | re.IGNORECASE
        )
        self.topic_shift_pattern = re.compile(
            '|'.join(self.TOPIC_SHIFT_PATTERNS),
            re.IGNORECASE
        )
        
        # Initialize tokenizer
        if tiktoken:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        else:
            self.tokenizer = None
            logger.warning("tiktoken not available, using word-based approximation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return int(len(text.split()) * 1.3)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key terms from text for hybrid search.
        
        Uses simple TF-based extraction without external dependencies.
        """
        # Clean and tokenize
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
            'were', 'being', 'their', 'there', 'this', 'that', 'with', 'they',
            'will', 'would', 'could', 'should', 'from', 'what', 'which', 'when',
            'where', 'who', 'how', 'than', 'then', 'these', 'those', 'into',
            'over', 'after', 'before', 'between', 'under', 'above', 'such',
            'each', 'some', 'other', 'only', 'also', 'more', 'most', 'very',
            'just', 'about', 'using', 'used', 'use', 'based', 'provide',
            'provides', 'including', 'included', 'include', 'within', 'through'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stopwords and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def detect_semantic_boundary(self, text: str, position: int) -> bool:
        """
        Detect if a position in text is a good semantic boundary.
        
        Looks for:
        - Paragraph breaks
        - Section headings
        - Topic shift indicators
        """
        # Check for double newline (paragraph break)
        if position > 0 and text[position-1:position+1] == '\n\n':
            return True
        
        # Get text around position
        start = max(0, position - 50)
        end = min(len(text), position + 50)
        context = text[start:end]
        
        # Check for boundary patterns
        if self.boundary_pattern.search(context):
            return True
        
        return False
    
    def find_best_split_point(
        self, 
        text: str, 
        target_pos: int, 
        window: int = 200
    ) -> int:
        """
        Find the best position to split text near target_pos.
        
        Prioritizes:
        1. Paragraph breaks (double newline)
        2. Section boundaries
        3. Sentence endings
        4. Clause boundaries (commas, semicolons)
        """
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        
        best_pos = target_pos
        best_score = 0
        
        for pos in range(start, end):
            score = 0
            
            # Paragraph break - highest priority
            if pos < len(text) - 1 and text[pos:pos+2] == '\n\n':
                score = 100
            # Single newline
            elif text[pos] == '\n':
                score = 50
            # Sentence ending
            elif pos < len(text) - 1 and text[pos] in '.!?' and text[pos+1] == ' ':
                score = 30
            # Clause boundary
            elif pos < len(text) - 1 and text[pos] in ',;:' and text[pos+1] == ' ':
                score = 10
            
            # Prefer positions closer to target
            distance_penalty = abs(pos - target_pos) / window * 20
            score -= distance_penalty
            
            if score > best_score:
                best_score = score
                best_pos = pos + 1  # Split after the boundary character
        
        return best_pos
    
    def chunk_with_sliding_window(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        document_type: str
    ) -> List[Tuple[str, int, int]]:
        """
        Chunk text using sliding window with semantic boundary awareness.
        
        Returns list of (chunk_text, start_char, end_char) tuples.
        """
        chunks = []
        text_len = len(text)
        
        if text_len == 0:
            return chunks
        
        # Convert token sizes to approximate character sizes
        char_per_token = 4  # Rough approximation
        target_chars = chunk_size * char_per_token
        overlap_chars = overlap * char_per_token
        
        start = 0
        chunk_index = 0
        
        while start < text_len:
            # Calculate target end position
            target_end = min(start + target_chars, text_len)
            
            # Find best split point
            if target_end < text_len:
                end = self.find_best_split_point(text, target_end)
            else:
                end = text_len
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            # Only add if meets minimum size
            if self.count_tokens(chunk_text) >= self.min_chunk_size or start == 0:
                chunks.append((chunk_text, start, end))
                chunk_index += 1
            
            # Move start position with overlap
            if end >= text_len:
                break
            
            # Calculate next start with overlap
            next_start = end - overlap_chars
            
            # Find good start point (preferably at sentence/paragraph boundary)
            next_start = self.find_best_split_point(text, next_start, window=100)
            
            # Ensure we make progress
            if next_start <= start:
                next_start = end
            
            start = next_start
        
        return chunks
    
    def chunk_criteria(
        self,
        chunks: List[Any],
        document_id: str = "criteria"
    ) -> List[TextChunk]:
        """
        Chunk a criteria/rubric document with KSB awareness.
        
        Strategy:
        - Keep each KSB criterion together when possible
        - Detect named sections like "Data (K1, K2):"
        - Preserve grade level descriptors
        - Extract keywords for hybrid search
        """
        result_chunks = []
        chunk_counter = 0
        
        current_criterion = None
        current_content = []
        current_tokens = 0
        current_page = 1
        parent_section = None
        
        # Patterns for detecting criteria and KSBs
        criterion_pattern = re.compile(
            r'(?:criterion|criteria|learning\s+outcome|LO)\s*(\d+)',
            re.IGNORECASE
        )
        
        ksb_pattern = re.compile(r'\b([KSB]\d{1,2})\b')
        
        named_criterion_pattern = re.compile(
            r'^((?:Part|Section)\s+\d+[:\s-]|[A-Z][a-z]+(?:\s+&\s+[A-Z][a-z]+)*\s*\([^)]*[KSB]\d+[^)]*\)\s*:)',
            re.IGNORECASE
        )
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            page = getattr(chunk, 'page_estimate', 1) or getattr(chunk, 'page_number', 1)
            chunk_type = getattr(chunk, 'chunk_type', 'text')
            
            # Track parent sections (headings)
            if chunk_type == 'heading':
                heading_level = getattr(chunk, 'heading_level', 1)
                if heading_level and heading_level <= 2:
                    parent_section = content[:100]
            
            # Check for named criterion pattern
            named_match = named_criterion_pattern.search(content)
            ksb_matches = ksb_pattern.findall(content)
            criterion_match = criterion_pattern.search(content)
            
            # Determine new criterion
            new_criterion = None
            if named_match:
                if ksb_matches:
                    new_criterion = ','.join(sorted(set(ksb_matches)))
                else:
                    new_criterion = named_match.group(1).strip().rstrip(':').rstrip('-')
            elif ksb_matches and len(ksb_matches) >= 1:
                new_criterion = ','.join(sorted(set(ksb_matches)))
            elif criterion_match:
                new_criterion = criterion_match.group(1)
            
            # Check if we should start a new criterion chunk
            should_start_new = (
                new_criterion and 
                new_criterion != current_criterion and
                (named_match or len(ksb_matches) >= 2)
            )
            
            if should_start_new:
                # Save previous criterion
                if current_content:
                    combined_content = '\n\n'.join(current_content)
                    keywords = self.extract_keywords(combined_content)
                    
                    result_chunks.append(TextChunk(
                        content=combined_content,
                        chunk_id=f"{document_id}_chunk_{chunk_counter}",
                        document_type='criteria',
                        token_count=self.count_tokens(combined_content),
                        page_start=current_page,
                        page_end=page,
                        criterion_id=current_criterion or 'general',
                        chunk_type='criterion',
                        keywords=keywords,
                        parent_section=parent_section,
                        chunk_index=chunk_counter
                    ))
                    chunk_counter += 1
                
                current_criterion = new_criterion
                current_content = [content]
                current_tokens = self.count_tokens(content)
                current_page = page
            else:
                content_tokens = self.count_tokens(content)
                
                if current_tokens + content_tokens > self.criteria_chunk_size:
                    # Save current and start new
                    if current_content:
                        combined_content = '\n\n'.join(current_content)
                        keywords = self.extract_keywords(combined_content)
                        
                        result_chunks.append(TextChunk(
                            content=combined_content,
                            chunk_id=f"{document_id}_chunk_{chunk_counter}",
                            document_type='criteria',
                            token_count=self.count_tokens(combined_content),
                            page_start=current_page,
                            page_end=page,
                            criterion_id=current_criterion or 'general',
                            chunk_type='criterion' if current_criterion else 'text',
                            keywords=keywords,
                            parent_section=parent_section,
                            chunk_index=chunk_counter
                        ))
                        chunk_counter += 1
                    
                    current_content = [content]
                    current_tokens = content_tokens
                    current_page = page
                else:
                    current_content.append(content)
                    current_tokens += content_tokens
        
        # Don't forget the last chunk
        if current_content:
            combined_content = '\n\n'.join(current_content)
            keywords = self.extract_keywords(combined_content)
            
            result_chunks.append(TextChunk(
                content=combined_content,
                chunk_id=f"{document_id}_chunk_{chunk_counter}",
                document_type='criteria',
                token_count=self.count_tokens(combined_content),
                page_start=current_page,
                page_end=current_page,
                criterion_id=current_criterion or 'general',
                chunk_type='criterion' if current_criterion else 'text',
                keywords=keywords,
                parent_section=parent_section,
                chunk_index=chunk_counter
            ))
        
        logger.info(f"Created {len(result_chunks)} criteria chunks")
        return result_chunks
    
    def chunk_report(
        self,
        chunks: List[Any],
        document_id: str = "report"
    ) -> List[TextChunk]:
        """
        Chunk a student report with semantic boundary awareness.
        
        Strategy:
        - Preserve section structure
        - Use sliding window with smart overlap
        - Keep tables intact
        - Extract keywords for hybrid search
        - Maintain figure references with context
        """
        result_chunks = []
        chunk_counter = 0
        
        current_section = None
        current_section_number = None
        parent_section = None
        current_content = []
        current_tokens = 0
        current_page_start = 1
        current_page_end = 1
        current_figures = []
        has_figure_ref = False
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            page = getattr(chunk, 'page_estimate', 1) or getattr(chunk, 'page_number', 1)
            chunk_type = getattr(chunk, 'chunk_type', 'text')
            heading_level = getattr(chunk, 'heading_level', None)
            figure_ids = getattr(chunk, 'figure_ids', [])
            has_fig = getattr(chunk, 'has_figure_reference', False)
            section_num = getattr(chunk, 'section_number', None)
            
            content_tokens = self.count_tokens(content)
            
            # Track figures
            if figure_ids:
                current_figures.extend(figure_ids)
                has_figure_ref = True
            
            # Track parent section for hierarchy
            is_heading = chunk_type == 'heading' or heading_level is not None
            if is_heading and heading_level and heading_level <= 2:
                parent_section = content[:100]
            
            # Determine if we should start a new chunk
            should_split = False
            
            # Split if too large
            if current_tokens + content_tokens > self.report_chunk_size:
                should_split = True
            # Split on major section boundaries
            elif is_heading and heading_level and heading_level <= 2 and current_content:
                should_split = True
            # Split if content would exceed max size
            elif current_tokens + content_tokens > self.max_chunk_size:
                should_split = True
            
            if should_split and current_content:
                combined_content = '\n\n'.join(current_content)
                keywords = self.extract_keywords(combined_content)
                
                result_chunks.append(TextChunk(
                    content=combined_content,
                    chunk_id=f"{document_id}_chunk_{chunk_counter}",
                    document_type='report',
                    token_count=self.count_tokens(combined_content),
                    page_start=current_page_start,
                    page_end=current_page_end,
                    section_title=current_section,
                    section_number=current_section_number,
                    has_figure_reference=has_figure_ref,
                    figure_ids=current_figures.copy(),
                    keywords=keywords,
                    parent_section=parent_section,
                    chunk_index=chunk_counter
                ))
                chunk_counter += 1
                
                # Start new chunk with overlap context
                overlap_content = []
                overlap_tokens = 0
                
                # Include last paragraph(s) for context overlap
                for prev_content in reversed(current_content):
                    prev_tokens = self.count_tokens(prev_content)
                    if overlap_tokens + prev_tokens <= self.report_overlap:
                        overlap_content.insert(0, prev_content)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                current_content = overlap_content + [content]
                current_tokens = overlap_tokens + content_tokens
                current_page_start = page
                current_figures = figure_ids.copy()
                has_figure_ref = has_fig
            else:
                if not current_content:
                    current_page_start = page
                
                current_content.append(content)
                current_tokens += content_tokens
            
            # Update section tracking
            if is_heading:
                current_section = content
                if section_num:
                    current_section_number = section_num
            
            current_page_end = page
        
        # Don't forget the last chunk
        if current_content:
            combined_content = '\n\n'.join(current_content)
            keywords = self.extract_keywords(combined_content)
            
            result_chunks.append(TextChunk(
                content=combined_content,
                chunk_id=f"{document_id}_chunk_{chunk_counter}",
                document_type='report',
                token_count=self.count_tokens(combined_content),
                page_start=current_page_start,
                page_end=current_page_end,
                section_title=current_section,
                section_number=current_section_number,
                has_figure_reference=has_figure_ref,
                figure_ids=current_figures,
                keywords=keywords,
                parent_section=parent_section,
                chunk_index=chunk_counter
            ))
        
        logger.info(f"Created {len(result_chunks)} report chunks")
        return result_chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunked document."""
        if not chunks:
            return {'total_chunks': 0}
        
        token_counts = [c.token_count for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens': sum(token_counts) / len(chunks),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'chunks_with_figures': sum(1 for c in chunks if c.has_figure_reference),
            'chunks_with_keywords': sum(1 for c in chunks if c.keywords),
            'document_type': chunks[0].document_type if chunks else None
        }