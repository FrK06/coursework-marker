"""
Smart Chunker - FIXED VERSION with better section detection and more granular chunks.
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
    document_type: str
    token_count: int
    
    page_start: int = 1
    page_end: int = 1
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    
    chunk_type: str = 'text'
    has_figure_reference: bool = False
    figure_ids: List[str] = field(default_factory=list)
    
    criterion_id: Optional[str] = None
    rubric_level: Optional[str] = None
    
    keywords: List[str] = field(default_factory=list)
    parent_section: Optional[str] = None
    chunk_index: int = 0
    
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
    """Smart document chunker with improved section awareness."""
    
    # FIXED: Reduced chunk sizes to create more chunks with better granularity
    def __init__(
        self,
        criteria_chunk_size: int = 300,  # Reduced from 400
        criteria_overlap: int = 50,
        report_chunk_size: int = 400,    # Reduced from 600
        report_overlap: int = 80,        # Reduced from 120
        min_chunk_size: int = 50,        # Reduced from 100
        max_chunk_size: int = 600,       # Reduced from 1000
        encoding_name: str = "cl100k_base"
    ):
        self.criteria_chunk_size = criteria_chunk_size
        self.criteria_overlap = criteria_overlap
        self.report_chunk_size = report_chunk_size
        self.report_overlap = report_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
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
        """Extract key terms from text for hybrid search."""
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        
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
        
        word_freq = {}
        for word in words:
            if word not in stopwords and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _detect_section_from_content(self, text: str) -> tuple:
        """
        FIXED: Detect section headers from content even without proper styles.
        
        Returns: (is_heading, section_number, heading_level)
        """
        text_stripped = text.strip()
        
        # Pattern 1: "Part X - Title" or "Part X: Title"
        part_match = re.match(r'^Part\s+(\d+)\s*[-:–]\s*(.+)$', text_stripped, re.IGNORECASE)
        if part_match:
            return True, part_match.group(1), 1
        
        # Pattern 2: "Section X - Title" or "Section X: Title"
        section_match = re.match(r'^Section\s+(\d+)\s*[-:–]\s*(.+)$', text_stripped, re.IGNORECASE)
        if section_match:
            return True, section_match.group(1), 1
        
        # Pattern 3: "1. Title" or "1.2 Title" or "1.2.3 Title"
        numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.\s*([A-Z].+)$', text_stripped)
        if numbered_match:
            number = numbered_match.group(1)
            level = number.count('.') + 1
            return True, number, level
        
        # Pattern 4: "1 Title" (number followed by capitalized word)
        simple_numbered = re.match(r'^(\d+)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)', text_stripped)
        if simple_numbered and len(text_stripped) < 80:
            return True, simple_numbered.group(1), 2
        
        # Pattern 5: Title case short lines (likely headings)
        if len(text_stripped) < 60 and text_stripped.istitle():
            words = text_stripped.split()
            # Exclude if it's a sentence (ends with period or contains common sentence words)
            if not text_stripped.endswith('.') and len(words) <= 8:
                return True, None, 2
        
        # Pattern 6: Common section titles
        common_headings = [
            'executive summary', 'introduction', 'conclusion', 'methodology',
            'requirements', 'results', 'discussion', 'references', 'appendix',
            'technical feasibility', 'proof of concept', 'benchmarking',
            'functional requirements', 'non-functional requirements',
            'data pipeline', 'model training', 'evaluation', 'deployment'
        ]
        if text_stripped.lower() in common_headings:
            return True, None, 2
        
        return False, None, None
    
    def find_best_split_point(self, text: str, target_pos: int, window: int = 200) -> int:
        """Find the best position to split text near target_pos."""
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        
        best_pos = target_pos
        best_score = 0
        
        for pos in range(start, end):
            score = 0
            
            if pos < len(text) - 1 and text[pos:pos+2] == '\n\n':
                score = 100
            elif text[pos] == '\n':
                score = 50
            elif pos < len(text) - 1 and text[pos] in '.!?' and text[pos+1] == ' ':
                score = 30
            elif pos < len(text) - 1 and text[pos] in ',;:' and text[pos+1] == ' ':
                score = 10
            
            distance_penalty = abs(pos - target_pos) / window * 20
            score -= distance_penalty
            
            if score > best_score:
                best_score = score
                best_pos = pos + 1
        
        return best_pos
    
    def chunk_criteria(self, chunks: List[Any], document_id: str = "criteria") -> List[TextChunk]:
        """Chunk a criteria/rubric document with KSB awareness."""
        result_chunks = []
        chunk_counter = 0
        
        current_criterion = None
        current_content = []
        current_tokens = 0
        current_page = 1
        parent_section = None
        
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
            
            # FIXED: Also detect sections from content
            is_heading, section_num, heading_level = self._detect_section_from_content(content)
            if is_heading and heading_level and heading_level <= 2:
                parent_section = content[:100]
            
            if chunk_type == 'heading':
                heading_level = getattr(chunk, 'heading_level', 1)
                if heading_level and heading_level <= 2:
                    parent_section = content[:100]
            
            named_match = named_criterion_pattern.search(content)
            ksb_matches = ksb_pattern.findall(content)
            criterion_match = criterion_pattern.search(content)
            
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
            
            should_start_new = (
                new_criterion and 
                new_criterion != current_criterion and
                (named_match or len(ksb_matches) >= 2)
            )
            
            if should_start_new:
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
    
    def chunk_report(self, chunks: List[Any], document_id: str = "report") -> List[TextChunk]:
        """
        FIXED: Chunk a student report with better semantic boundary awareness.
        Creates more granular chunks to preserve document structure.
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
            
            # FIXED: Detect sections from content if not already a heading
            is_content_heading = False
            content_section_num = None
            content_heading_level = None
            
            if chunk_type != 'heading':
                is_content_heading, content_section_num, content_heading_level = self._detect_section_from_content(content)
                if is_content_heading:
                    chunk_type = 'heading'
                    heading_level = content_heading_level
                    section_num = content_section_num
            
            content_tokens = self.count_tokens(content)
            
            if figure_ids:
                current_figures.extend(figure_ids)
                has_figure_ref = True
            
            is_heading = chunk_type == 'heading' or heading_level is not None
            if is_heading and heading_level and heading_level <= 2:
                parent_section = content[:100]
            
            should_split = False
            
            # FIXED: More aggressive splitting for better granularity
            if current_tokens + content_tokens > self.report_chunk_size:
                should_split = True
            elif is_heading and current_content:  # Split on ANY heading
                should_split = True
            elif current_tokens + content_tokens > self.max_chunk_size:
                should_split = True
            # FIXED: Also split on paragraph breaks if chunk is getting long
            elif current_tokens > self.report_chunk_size * 0.7 and content.startswith('\n'):
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
                
                # FIXED: Smaller overlap to create more distinct chunks
                overlap_content = []
                overlap_tokens = 0
                
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
            
            if is_heading:
                current_section = content
                if section_num:
                    current_section_number = section_num
            
            current_page_end = page
        
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
        
        logger.info(f"Created {len(result_chunks)} report chunks (FIXED chunker)")
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
            'chunks_with_sections': sum(1 for c in chunks if c.section_title),
            'document_type': chunks[0].document_type if chunks else None
        }
