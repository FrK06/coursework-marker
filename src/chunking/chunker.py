"""
Smart Chunker - Structure-aware document chunking.

Different chunking strategies for:
- Criteria documents: Smaller chunks (300-600 tokens) for precise matching
- Student reports: Larger chunks (600-900 tokens) for context preservation

Preserves:
- Section structure and headings
- Table integrity
- Figure references and context
"""
import re
from typing import List, Dict, Any, Optional
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
        }


class SmartChunker:
    """
    Smart document chunker with different strategies for different document types.
    
    Criteria Documents:
    - Smaller chunks (300-600 tokens)
    - Preserve individual criteria/rubric items
    - Keep grade descriptors together
    
    Student Reports:
    - Larger chunks (600-900 tokens)
    - Preserve section structure
    - Keep tables intact
    - Maintain figure references with context
    """
    
    def __init__(
        self,
        criteria_chunk_size: int = 400,
        criteria_overlap: int = 50,
        report_chunk_size: int = 700,
        report_overlap: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.
        
        Args:
            criteria_chunk_size: Target tokens for criteria chunks
            criteria_overlap: Overlap tokens for criteria
            report_chunk_size: Target tokens for report chunks
            report_overlap: Overlap tokens for reports
            encoding_name: Tiktoken encoding (cl100k_base for most models)
        """
        self.criteria_chunk_size = criteria_chunk_size
        self.criteria_overlap = criteria_overlap
        self.report_chunk_size = report_chunk_size
        self.report_overlap = report_overlap
        
        # Initialize tokenizer
        if tiktoken:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        else:
            # Fallback: approximate tokens as words * 1.3
            self.tokenizer = None
            logger.warning("tiktoken not available, using word-based approximation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: ~1.3 tokens per word
            return int(len(text.split()) * 1.3)
    
    def chunk_criteria(
        self,
        chunks: List[Any],
        document_id: str = "criteria"
    ) -> List[TextChunk]:
        """
        Chunk a criteria/rubric document.
        
        Strategy:
        - Try to keep each criterion as a single chunk
        - Detect KSBs (K1, K2, S1, S16, B1, etc.) as criterion IDs
        - Detect named sections like "Data (K1, K2):", "Model (K1):", etc.
        - Split large criteria by grade level if possible
        - Preserve rubric tables
        """
        result_chunks = []
        chunk_counter = 0
        
        current_criterion = None
        current_criterion_name = None
        current_content = []
        current_tokens = 0
        current_page = 1
        
        # Patterns for detecting criteria and KSBs
        criterion_pattern = re.compile(
            r'(?:criterion|criteria|learning\s+outcome|LO)\s*(\d+)',
            re.IGNORECASE
        )
        
        # KSB pattern: K1, K2, K16, S1, S16, B1, etc.
        ksb_pattern = re.compile(
            r'\b([KSB]\d{1,2})\b'
        )
        
        # Named criterion pattern: "Data (K1, K2, K18, S16):" or "Model (K1, K18):"
        # Also matches "Part 1:", "Part 2:", "Section 1:", etc.
        named_criterion_pattern = re.compile(
            r'^((?:Part|Section)\s+\d+[:\s-]|[A-Z][a-z]+(?:\s+&\s+[A-Z][a-z]+)*\s*\([^)]*[KSB]\d+[^)]*\)\s*:)',
            re.IGNORECASE
        )
        
        rubric_levels = ['distinction', 'merit', 'pass', 'fail', 'refer']
        
        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            page = getattr(chunk, 'page_estimate', 1) or getattr(chunk, 'page_number', 1)
            chunk_type = getattr(chunk, 'chunk_type', 'text')
            
            # Check for named criterion pattern first (e.g., "Data (K1, K2, K18, S16):")
            named_match = named_criterion_pattern.search(content)
            
            # Check for KSBs
            ksb_matches = ksb_pattern.findall(content)
            
            # Check for numbered criterion
            criterion_match = criterion_pattern.search(content)
            
            # Determine if this starts a new criterion
            new_criterion = None
            new_criterion_name = None
            
            if named_match:
                # Extract criterion name and KSBs
                new_criterion_name = named_match.group(1).strip().rstrip(':').rstrip('-')
                if ksb_matches:
                    new_criterion = ','.join(sorted(set(ksb_matches)))
                else:
                    new_criterion = new_criterion_name
            elif ksb_matches and len(ksb_matches) >= 1:
                # Multiple KSBs suggest a new criterion section
                new_criterion = ','.join(sorted(set(ksb_matches)))
            elif criterion_match:
                new_criterion = criterion_match.group(1)
            
            # Check if we should start a new criterion chunk
            should_start_new = (
                new_criterion and 
                new_criterion != current_criterion and
                # Only start new if we have meaningful content or it's a named section
                (named_match or len(ksb_matches) >= 2)
            )
            
            if should_start_new:
                # Save previous criterion
                if current_content:
                    result_chunks.append(self._create_chunk(
                        content='\n\n'.join(current_content),
                        chunk_id=f"{document_id}_chunk_{chunk_counter}",
                        document_type='criteria',
                        page_start=current_page,
                        page_end=page,
                        criterion_id=current_criterion or 'general',
                        chunk_type='criterion'
                    ))
                    chunk_counter += 1
                
                # Start new criterion
                current_criterion = new_criterion
                current_criterion_name = new_criterion_name                                 #"current_criterion_name" is not accessed
                current_content = [content]
                current_tokens = self.count_tokens(content)
                current_page = page
            
            else:
                # Check if we need to split due to size
                content_tokens = self.count_tokens(content)
                
                if current_tokens + content_tokens > self.criteria_chunk_size:
                    # Save current and start new
                    if current_content:
                        result_chunks.append(self._create_chunk(
                            content='\n\n'.join(current_content),
                            chunk_id=f"{document_id}_chunk_{chunk_counter}",
                            document_type='criteria',
                            page_start=current_page,
                            page_end=page,
                            criterion_id=current_criterion or 'general',
                            chunk_type='criterion' if current_criterion else 'text'
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
            result_chunks.append(self._create_chunk(
                content='\n\n'.join(current_content),
                chunk_id=f"{document_id}_chunk_{chunk_counter}",
                document_type='criteria',
                page_start=current_page,
                page_end=current_page,
                criterion_id=current_criterion or 'general',
                chunk_type='criterion' if current_criterion else 'text'
            ))
        
        return result_chunks
    
    def chunk_report(
        self,
        chunks: List[Any],
        document_id: str = "report"
    ) -> List[TextChunk]:
        """
        Chunk a student report document.
        
        Strategy:
        - Preserve section structure (keep headings with their content)
        - Keep tables intact (don't split mid-table)
        - Maintain figure references with surrounding context
        - Use larger chunks with overlap
        """
        result_chunks = []
        chunk_counter = 0
        
        current_section = None
        current_section_number = None
        current_content = []
        current_tokens = 0
        current_page_start = 1
        current_page_end = 1
        current_figures = []
        has_figure_ref = False
        
        for i, chunk in enumerate(chunks):                                                                      #  "i" is not accessed        
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
            
            # Check if this is a heading (potential section boundary)
            is_heading = chunk_type == 'heading' or heading_level is not None
            
            # Determine if we should start a new chunk
            should_split = False
            
            # Split if: too large, or new major section, or table that would exceed limit
            if current_tokens + content_tokens > self.report_chunk_size:
                should_split = True
            elif is_heading and heading_level and heading_level <= 2 and current_content:
                # New major section (h1 or h2)
                should_split = True
            
            if should_split and current_content:
                # Create chunk with overlap
                chunk_content = '\n\n'.join(current_content)
                
                result_chunks.append(self._create_chunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{chunk_counter}",
                    document_type='report',
                    page_start=current_page_start,
                    page_end=current_page_end,
                    section_title=current_section,
                    section_number=current_section_number,
                    has_figure_reference=has_figure_ref,
                    figure_ids=current_figures.copy()
                ))
                chunk_counter += 1
                
                # Start new chunk with overlap
                # Include last item for context if it wasn't a heading
                overlap_content = []
                if current_content and not is_heading:
                    last_content = current_content[-1]
                    last_tokens = self.count_tokens(last_content)
                    if last_tokens <= self.report_overlap:
                        overlap_content = [last_content]
                
                current_content = overlap_content + [content]
                current_tokens = sum(self.count_tokens(c) for c in current_content)
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
            result_chunks.append(self._create_chunk(
                content='\n\n'.join(current_content),
                chunk_id=f"{document_id}_chunk_{chunk_counter}",
                document_type='report',
                page_start=current_page_start,
                page_end=current_page_end,
                section_title=current_section,
                section_number=current_section_number,
                has_figure_reference=has_figure_ref,
                figure_ids=current_figures
            ))
        
        return result_chunks
    
    def _create_chunk(
        self,
        content: str,
        chunk_id: str,
        document_type: str,
        page_start: int = 1,
        page_end: int = 1,
        section_title: Optional[str] = None,
        section_number: Optional[str] = None,
        criterion_id: Optional[str] = None,
        chunk_type: str = 'text',
        has_figure_reference: bool = False,
        figure_ids: List[str] = None
    ) -> TextChunk:
        """Create a TextChunk with all metadata."""
        return TextChunk(
            content=content,
            chunk_id=chunk_id,
            document_type=document_type,
            token_count=self.count_tokens(content),
            page_start=page_start,
            page_end=page_end,
            section_title=section_title,
            section_number=section_number,
            chunk_type=chunk_type,
            has_figure_reference=has_figure_reference,
            figure_ids=figure_ids or [],
            criterion_id=criterion_id
        )
    
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
            'document_type': chunks[0].document_type if chunks else None
        }
