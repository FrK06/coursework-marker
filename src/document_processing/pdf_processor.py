"""
PDF Processor - Extracts text content from PDF documents.

Primarily used for marking criteria/rubric documents which are often in PDF format.
Uses pdfplumber for robust text extraction including tables.
"""
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Represents a chunk of PDF content with metadata."""
    content: str
    chunk_type: str  # 'text', 'table', 'heading'
    page_number: int
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedPDF:
    """Complete extracted PDF document."""
    chunks: List[PDFChunk]
    title: Optional[str] = None
    total_pages: int = 0
    raw_text: str = ""


class PDFProcessor:
    """
    Processes PDF files to extract text content.
    
    Designed primarily for marking criteria/rubric documents.
    Extracts:
    - Text content with page numbers
    - Tables (converted to text format)
    - Attempts to detect headings/sections
    """
    
    # Patterns that might indicate a heading
    HEADING_PATTERNS = [
        r'^(?:criterion|criteria)\s*\d*[:.]?\s*',
        r'^(?:learning\s+outcome)\s*\d*[:.]?\s*',
        r'^(?:assessment\s+criteria)\s*',
        r'^\d+\.\s+[A-Z]',  # Numbered sections starting with capital
        r'^[A-Z][A-Z\s]+$',  # ALL CAPS lines
        r'^(?:pass|merit|distinction|fail)[:.]?\s*',
    ]
    
    def __init__(self):
        if pdfplumber is None:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
        
        self.heading_pattern = re.compile(
            '|'.join(self.HEADING_PATTERNS), 
            re.IGNORECASE
        )
    
    def process(self, file_path: str) -> ExtractedPDF:
        """
        Process a PDF file and extract all content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractedPDF with all extracted content and metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"Expected .pdf file, got: {path.suffix}")
        
        chunks = []
        raw_text_parts = []
        current_section = None
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract tables first
                tables = page.extract_tables()
                table_texts = set()
                
                for table in tables:
                    table_chunk = self._process_table(
                        table, 
                        page_num, 
                        current_section
                    )
                    if table_chunk:
                        chunks.append(table_chunk)
                        raw_text_parts.append(table_chunk.content)
                        # Track table text to avoid duplication
                        for row in table:
                            for cell in row:
                                if cell:
                                    table_texts.add(cell.strip())
                
                # Extract text
                text = page.extract_text() or ""
                
                # Process text line by line
                lines = text.split('\n')
                current_paragraph = []
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip if this text was in a table
                    if line in table_texts:
                        continue
                    
                    if not line:
                        # End of paragraph
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            chunk = self._create_text_chunk(
                                para_text,
                                page_num,
                                current_section
                            )
                            if chunk:
                                # Update section tracking
                                if chunk.chunk_type == 'heading':
                                    current_section = chunk.content
                                chunks.append(chunk)
                                raw_text_parts.append(chunk.content)
                            current_paragraph = []
                    else:
                        # Check if this line is a heading
                        if self._is_heading(line):
                            # Save current paragraph first
                            if current_paragraph:
                                para_text = ' '.join(current_paragraph)
                                chunk = self._create_text_chunk(
                                    para_text,
                                    page_num,
                                    current_section
                                )
                                if chunk:
                                    chunks.append(chunk)
                                    raw_text_parts.append(chunk.content)
                                current_paragraph = []
                            
                            # Create heading chunk
                            heading_chunk = PDFChunk(
                                content=line,
                                chunk_type='heading',
                                page_number=page_num,
                                section_title=line
                            )
                            chunks.append(heading_chunk)
                            raw_text_parts.append(line)
                            current_section = line
                        else:
                            current_paragraph.append(line)
                
                # Handle remaining paragraph
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    chunk = self._create_text_chunk(
                        para_text,
                        page_num,
                        current_section
                    )
                    if chunk:
                        chunks.append(chunk)
                        raw_text_parts.append(chunk.content)
        
        # Try to extract title
        title = self._extract_title(chunks)
        
        return ExtractedPDF(
            chunks=chunks,
            title=title,
            total_pages=total_pages,
            raw_text="\n\n".join(raw_text_parts)
        )
    
    def _is_heading(self, line: str) -> bool:
        """Check if a line appears to be a heading."""
        # Check against patterns
        if self.heading_pattern.match(line):
            return True
        
        # Short lines in title case might be headings
        if len(line) < 100 and line.istitle():
            return True
        
        # All caps short lines
        if len(line) < 80 and line.isupper() and len(line.split()) > 1:
            return True
        
        return False
    
    def _create_text_chunk(
        self,
        text: str,
        page_number: int,
        current_section: Optional[str]
    ) -> Optional[PDFChunk]:
        """Create a text chunk from paragraph text."""
        text = text.strip()
        if not text or len(text) < 10:
            return None
        
        # Check if it's actually a heading
        if self._is_heading(text) and len(text) < 150:
            return PDFChunk(
                content=text,
                chunk_type='heading',
                page_number=page_number,
                section_title=text
            )
        
        return PDFChunk(
            content=text,
            chunk_type='text',
            page_number=page_number,
            section_title=current_section
        )
    
    def _process_table(
        self,
        table: List[List[str]],
        page_number: int,
        current_section: Optional[str]
    ) -> Optional[PDFChunk]:
        """Convert a table to readable text format."""
        if not table or not table[0]:
            return None
        
        # Filter out empty rows
        table = [row for row in table if any(cell for cell in row)]
        
        if not table:
            return None
        
        # Build markdown-style table
        markdown_rows = []
        
        # Header row
        header = [str(cell or '').strip() for cell in table[0]]
        markdown_rows.append("| " + " | ".join(header) + " |")
        markdown_rows.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in table[1:]:
            cells = [str(cell or '').strip().replace('\n', ' ') for cell in row]
            # Pad if necessary
            while len(cells) < len(header):
                cells.append('')
            cells = cells[:len(header)]  # Truncate if too many
            markdown_rows.append("| " + " | ".join(cells) + " |")
        
        table_text = "\n".join(markdown_rows)
        
        return PDFChunk(
            content=table_text,
            chunk_type='table',
            page_number=page_number,
            section_title=current_section,
            metadata={'row_count': len(table), 'col_count': len(header)}
        )
    
    def _extract_title(self, chunks: List[PDFChunk]) -> Optional[str]:
        """Try to extract the document title from first page."""
        for chunk in chunks:
            if chunk.page_number == 1:
                if chunk.chunk_type == 'heading':
                    return chunk.content
        return None
    
    def extract_criteria_sections(self, doc: ExtractedPDF) -> List[Dict[str, Any]]:
        """
        Extract individual criteria/learning outcomes from a rubric document.
        
        Returns list of criteria with their descriptions and any grade descriptors.
        """
        criteria = []
        current_criterion = None
        
        criterion_pattern = re.compile(
            r'^(?:criterion|criteria|learning\s+outcome|LO)\s*(\d+)',
            re.IGNORECASE
        )
        
        for chunk in doc.chunks:
            match = criterion_pattern.match(chunk.content)
            
            if match:
                # Save previous criterion
                if current_criterion:
                    criteria.append(current_criterion)
                
                # Start new criterion
                current_criterion = {
                    'id': match.group(1),
                    'title': chunk.content,
                    'description': '',
                    'descriptors': [],
                    'page': chunk.page_number
                }
            elif current_criterion:
                # Add content to current criterion
                if chunk.chunk_type == 'table':
                    # Tables often contain grade descriptors
                    current_criterion['descriptors'].append(chunk.content)
                else:
                    current_criterion['description'] += '\n' + chunk.content
        
        # Don't forget last criterion
        if current_criterion:
            criteria.append(current_criterion)
        
        return criteria
