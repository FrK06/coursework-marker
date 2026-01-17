"""
Document Processing Module - Handles extraction from DOCX and PDF files.
"""
from .docx_processor import DocxProcessor
from .pdf_processor import PDFProcessor
from .image_extractor import ImageExtractor

__all__ = ["DocxProcessor", "PDFProcessor", "ImageExtractor"]
