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

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
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
        Chunk a student report with subsection merging and parent context.

        Requirements:
        1. Minimum 200 words per chunk (merge small subsections)
        2. Add parent section context to each chunk (prepended)
        3. Target 15-20 chunks (200-400 words each) for typical 3000-5000 word reports
        4. Skip chunks <50 words (except tables — tables are always kept)
        """
        result_chunks = []
        table_chunks = []  # Collect table chunks separately
        chunk_counter = 0

        # Track current parent section (level 1 heading)
        parent_section = None
        parent_section_title = None

        # Track current subsection
        current_section = None
        current_section_number = None

        # Accumulate content under same parent until minimum word count
        content_accumulator = []
        words_accumulated = 0
        current_page_start = 1
        current_page_end = 1
        current_figures = []
        has_figure_ref = False

        def create_chunk():
            """Helper to create a chunk from accumulated content."""
            nonlocal chunk_counter, content_accumulator, words_accumulated
            nonlocal current_page_start, current_figures, has_figure_ref

            if not content_accumulator:
                return

            merged_content = '\n\n'.join(content_accumulator)
            merged_words = self.count_words(merged_content)

            # Skip chunks < 50 words
            if merged_words < 50:
                logger.debug(f"Skipping chunk with only {merged_words} words")
                return

            # Prepend structured header with section number and parent context
            section_num_display = current_section_number if current_section_number else "?"
            parent_title_display = parent_section_title if parent_section_title else current_section

            if parent_title_display:
                # Structured header format: [Section X.X | Parent Title]
                header = f"[Section {section_num_display} | {parent_title_display}]"
                merged_content = f"{header}\n\n{merged_content}"

            keywords = self.extract_keywords(merged_content)

            result_chunks.append(TextChunk(
                content=merged_content,
                chunk_id=f"{document_id}_chunk_{chunk_counter}",
                document_type='report',
                token_count=self.count_tokens(merged_content),
                page_start=current_page_start,
                page_end=current_page_end,
                section_title=current_section,
                section_number=current_section_number,
                chunk_type='prose',
                has_figure_reference=has_figure_ref,
                figure_ids=current_figures.copy(),
                keywords=keywords,
                parent_section=parent_section,
                chunk_index=chunk_counter
            ))
            chunk_counter += 1

        def create_table_chunk(content: str, page: int, row_count: int, col_count: int):
            """Helper to create a dedicated table chunk with metadata header."""
            nonlocal chunk_counter

            # Build metadata header
            section_display = current_section or parent_section_title or "Unknown"
            # Try to extract a caption from the first row header
            lines = content.strip().split('\n')
            header_row = lines[0] if lines else ""
            caption = header_row.strip('| ').replace(' | ', ', ')[:80] if header_row else ""

            header = f"[TABLE in {section_display} | {row_count} rows × {col_count} cols"
            if caption:
                header += f" | {caption}"
            header += "]"

            table_content = f"{header}\n{content}"
            keywords = self.extract_keywords(table_content)

            table_chunks.append(TextChunk(
                content=table_content,
                chunk_id=f"{document_id}_table_{chunk_counter}",
                document_type='report',
                token_count=self.count_tokens(table_content),
                page_start=page,
                page_end=page,
                section_title=current_section,
                section_number=current_section_number,
                chunk_type='table',
                keywords=keywords,
                parent_section=parent_section,
                chunk_index=chunk_counter,
                metadata={'row_count': row_count, 'col_count': col_count}
            ))
            chunk_counter += 1

        for chunk in chunks:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            page = getattr(chunk, 'page_estimate', 1) or getattr(chunk, 'page_number', 1)
            chunk_type = getattr(chunk, 'chunk_type', 'text')
            heading_level = getattr(chunk, 'heading_level', None)
            figure_ids = getattr(chunk, 'figure_ids', [])
            has_fig = getattr(chunk, 'has_figure_reference', False)
            section_num = getattr(chunk, 'section_number', None)

            # Handle table chunks as dedicated first-class chunks
            if chunk_type == 'table':
                # Flush any accumulated prose before the table
                create_chunk()
                content_accumulator = []
                words_accumulated = 0
                current_page_start = page
                current_figures = []
                has_figure_ref = False

                # Extract row/col counts from metadata
                chunk_meta = getattr(chunk, 'metadata', {}) or {}
                row_count = chunk_meta.get('row_count', content.count('\n') + 1)
                col_count = chunk_meta.get('col_count', (content.split('\n')[0].count('|') - 1) if '|' in content else 0)
                create_table_chunk(content, page, row_count, col_count)
                continue

            # Detect sections from content if not already a heading
            if chunk_type != 'heading':
                is_content_heading, content_section_num, content_heading_level = self._detect_section_from_content(content)
                if is_content_heading:
                    chunk_type = 'heading'
                    heading_level = content_heading_level
                    section_num = content_section_num

            content_words = self.count_words(content)

            if figure_ids:
                current_figures.extend(figure_ids)
            if has_fig:
                has_figure_ref = True

            is_heading = chunk_type == 'heading' or heading_level is not None

            # Handle parent section (level 1 heading)
            if is_heading and heading_level and heading_level == 1:
                # New parent section - flush accumulated content first
                create_chunk()

                # Reset accumulator
                content_accumulator = []
                words_accumulated = 0
                current_page_start = page
                current_figures = []
                has_figure_ref = False

                # Update parent tracking
                parent_section = content[:100]
                parent_section_title = content.strip()
                current_section = content
                current_section_number = section_num
                current_page_end = page

                # Don't add parent heading to accumulator
                continue

            # Handle subsection headings (level 2+)
            if is_heading and heading_level and heading_level > 1:
                current_section = content
                if section_num:
                    current_section_number = section_num

            # Add content to accumulator
            content_accumulator.append(content)
            words_accumulated += content_words
            current_page_end = page

            # Check if we should create a chunk
            should_create = False

            # Condition 1: Reached target word count (200-400 words)
            if words_accumulated >= 200:
                # Check if next chunk would be a level-1 heading
                # If so, wait for it to trigger the flush
                should_create = True

            # Condition 2: Exceeded max chunk size (400 words)
            if words_accumulated >= 400:
                should_create = True

            if should_create:
                create_chunk()

                # Reset accumulator
                content_accumulator = []
                words_accumulated = 0
                current_page_start = page
                current_figures = []
                has_figure_ref = False

        # Flush remaining content
        create_chunk()

        # Combine prose and table chunks, sorted by chunk_index to preserve document order
        all_chunks = result_chunks + table_chunks
        all_chunks.sort(key=lambda c: c.chunk_index)

        prose_count = len(result_chunks)
        table_count = len(table_chunks)
        total_words = sum(self.count_words(c.content) for c in all_chunks)
        avg_words = total_words // max(len(all_chunks), 1)
        logger.info(f"Created {len(all_chunks)} report chunks ({prose_count} prose + {table_count} tables, avg {avg_words} words/chunk)")
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunked document."""
        if not chunks:
            return {'total_chunks': 0}

        token_counts = [c.token_count for c in chunks]
        word_counts = [self.count_words(c.content) for c in chunks]

        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens': sum(token_counts) / len(chunks),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_words': sum(word_counts),
            'avg_words': sum(word_counts) / len(chunks),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'chunks_with_figures': sum(1 for c in chunks if c.has_figure_reference),
            'chunks_with_keywords': sum(1 for c in chunks if c.keywords),
            'chunks_with_sections': sum(1 for c in chunks if c.section_title),
            'document_type': chunks[0].document_type if chunks else None
        }
