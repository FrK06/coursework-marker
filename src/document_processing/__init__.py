"""
Document Processing Module - Handles extraction from DOCX and PDF files.
"""
from .docx_processor import DocxProcessor
from .pdf_processor import PDFProcessor
from .image_processor import ImageProcessor, ProcessedImage

__all__ = ["DocxProcessor", "PDFProcessor", "ImageProcessor", "ProcessedImage"]
