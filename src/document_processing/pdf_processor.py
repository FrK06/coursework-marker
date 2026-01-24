"""
PDF Processor - Extracts text content from PDF documents.
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
    chunk_type: str
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
    source_type: str = "pdf"  # PDF has accurate page numbers
    pages_are_accurate: bool = True  # True for PDF


class PDFProcessor:
    """Processes PDF files to extract text content."""
    
    HEADING_PATTERNS = [
        r'^(?:criterion|criteria)\s*\d*[:.]?\s*',
        r'^(?:learning\s+outcome)\s*\d*[:.]?\s*',
        r'^(?:assessment\s+criteria)\s*',
        r'^\d+\.\s+[A-Z]',
        r'^[A-Z][A-Z\s]+$',
        r'^(?:pass|merit|distinction|fail)[:.]?\s*',
    ]
    
    def __init__(self):
        if pdfplumber is None:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
        
        self.heading_pattern = re.compile(
            '|'.join(self.HEADING_PATTERNS), re.IGNORECASE
        )
    
    def process(self, file_path: str) -> ExtractedPDF:
        """Process a PDF file and extract all content."""
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
                tables = page.extract_tables()
                table_texts = set()
                
                for table in tables:
                    table_chunk = self._process_table(table, page_num, current_section)
                    if table_chunk:
                        chunks.append(table_chunk)
                        raw_text_parts.append(table_chunk.content)
                        for row in table:
                            for cell in row:
                                if cell:
                                    table_texts.add(cell.strip())
                
                text = page.extract_text() or ""
                lines = text.split('\n')
                current_paragraph = []
                
                for line in lines:
                    line = line.strip()
                    
                    if line in table_texts:
                        continue
                    
                    if not line:
                        if current_paragraph:
                            para_text = ' '.join(current_paragraph)
                            chunk = self._create_text_chunk(para_text, page_num, current_section)
                            if chunk:
                                if chunk.chunk_type == 'heading':
                                    current_section = chunk.content
                                chunks.append(chunk)
                                raw_text_parts.append(chunk.content)
                            current_paragraph = []
                    else:
                        if self._is_heading(line):
                            if current_paragraph:
                                para_text = ' '.join(current_paragraph)
                                chunk = self._create_text_chunk(para_text, page_num, current_section)
                                if chunk:
                                    chunks.append(chunk)
                                    raw_text_parts.append(chunk.content)
                                current_paragraph = []
                            
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
                
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    chunk = self._create_text_chunk(para_text, page_num, current_section)
                    if chunk:
                        chunks.append(chunk)
                        raw_text_parts.append(chunk.content)
        
        title = self._extract_title(chunks)
        
        return ExtractedPDF(
            chunks=chunks,
            title=title,
            total_pages=total_pages,
            raw_text="\n\n".join(raw_text_parts)
        )
    
    def _is_heading(self, line: str) -> bool:
        if self.heading_pattern.match(line):
            return True
        if len(line) < 100 and line.istitle():
            return True
        if len(line) < 80 and line.isupper() and len(line.split()) > 1:
            return True
        return False
    
    def _create_text_chunk(
        self, text: str, page_number: int, current_section: Optional[str]
    ) -> Optional[PDFChunk]:
        text = text.strip()
        if not text or len(text) < 10:
            return None
        
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
        self, table: List[List[str]], page_number: int, current_section: Optional[str]
    ) -> Optional[PDFChunk]:
        if not table or not table[0]:
            return None
        
        table = [row for row in table if any(cell for cell in row)]
        if not table:
            return None
        
        markdown_rows = []
        header = [str(cell or '').strip() for cell in table[0]]
        markdown_rows.append("| " + " | ".join(header) + " |")
        markdown_rows.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        for row in table[1:]:
            cells = [str(cell or '').strip().replace('\n', ' ') for cell in row]
            while len(cells) < len(header):
                cells.append('')
            cells = cells[:len(header)]
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
        for chunk in chunks:
            if chunk.page_number == 1 and chunk.chunk_type == 'heading':
                return chunk.content
        return None
