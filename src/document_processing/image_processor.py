"""
Image Processor - Extracts and prepares images for vision model analysis.

Handles:
- Extracting images from DOCX files
- Extracting images from PDF files  
- Resizing for optimal model input
- Base64 encoding for Ollama vision API
"""
import io
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """An image ready for vision model analysis."""
    image_id: str
    base64_data: str
    format: str  # 'png', 'jpeg'
    width: int
    height: int
    caption: Optional[str] = None
    page: int = 1
    context: Optional[str] = None  # Surrounding text


class ImageProcessor:
    """
    Processes images from documents for vision model analysis.
    
    Extracts images from DOCX and PDF, resizes them appropriately,
    and prepares them for the Ollama vision API.
    """
    
    SUPPORTED_FORMATS = {'png', 'jpeg', 'jpg', 'gif', 'webp', 'bmp'}

    def __init__(self, max_size: Tuple[int, int] = (1024, 1024), ollama_client=None):
        """
        Initialize the image processor.

        Args:
            max_size: Maximum (width, height) for images sent to vision model
            ollama_client: Optional OllamaClient for OCR capabilities
        """
        if Image is None:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        self.max_size = max_size
        self.ollama_client = ollama_client

    def extract_text_with_ocr(self, image: ProcessedImage) -> str:
        """
        Extract text from image using GLM-OCR.

        Args:
            image: ProcessedImage with base64_data

        Returns:
            Extracted text, or empty string if OCR fails/disabled
        """
        from config import OCR_ENABLED

        if not OCR_ENABLED or not self.ollama_client:
            return ""

        if not hasattr(image, 'base64_data') or not image.base64_data:
            logger.debug(f"Image {getattr(image, 'image_id', 'unknown')} has no base64_data")
            return ""

        try:
            ocr_text = self.ollama_client.extract_text_from_image(
                image.base64_data,
                prompt="Extract all text, code, labels, and metrics from this image. "
                       "Preserve formatting and structure."
            )
            return ocr_text.strip()

        except Exception as e:
            image_id = getattr(image, 'image_id', 'unknown')
            logger.warning(f"OCR failed for {image_id}: {e}")
            return ""
    
    def process_docx_images(
        self,
        figures: Dict[str, bytes],
        captions: Dict[str, str]
    ) -> List[ProcessedImage]:
        """
        Process images extracted from a DOCX file.
        
        Args:
            figures: Dict of figure_id -> raw image bytes (from DocxProcessor)
            captions: Dict of figure_id -> caption text
            
        Returns:
            List of ProcessedImage ready for vision model
        """
        processed = []
        
        for figure_id, raw_bytes in figures.items():
            try:
                image = self._process_image_bytes(raw_bytes, figure_id)
                if image:
                    image.caption = captions.get(figure_id)
                    processed.append(image)
            except Exception as e:
                logger.warning(f"Could not process image {figure_id}: {e}")
        
        logger.info(f"Processed {len(processed)} images from DOCX")
        return processed
    
    def process_pdf_images(self, pdf_path: str) -> List[ProcessedImage]:
        """
        Extract and process images from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of ProcessedImage ready for vision model
        """
        if pdfplumber is None:
            logger.warning("pdfplumber not available for PDF image extraction")
            return []
        
        processed = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract images from page
                    if hasattr(page, 'images'):
                        for i, img_info in enumerate(page.images):
                            try:
                                # pdfplumber provides image data differently
                                # Try to extract the actual image
                                image_id = f"pdf_page{page_num}_img{i}"
                                
                                # Get image from page
                                if 'stream' in img_info:
                                    raw_bytes = img_info['stream'].get_data()
                                    image = self._process_image_bytes(raw_bytes, image_id)
                                    if image:
                                        image.page = page_num
                                        processed.append(image)
                            except Exception as e:
                                logger.debug(f"Could not extract image from PDF page {page_num}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting images from PDF: {e}")
        
        logger.info(f"Processed {len(processed)} images from PDF")
        return processed
    
    def _process_image_bytes(
        self,
        raw_bytes: bytes,
        image_id: str
    ) -> Optional[ProcessedImage]:
        """Process raw image bytes into a format suitable for vision model."""
        try:
            # Open image
            img = Image.open(io.BytesIO(raw_bytes))
            
            # Get format
            img_format = (img.format or 'PNG').lower()
            if img_format == 'jpeg':
                output_format = 'JPEG'
            else:
                output_format = 'PNG'
                img_format = 'png'
            
            # Convert palette/RGBA to RGB for JPEG
            if output_format == 'JPEG' and img.mode in ('RGBA', 'P', 'LA'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize if necessary
            if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            output = io.BytesIO()
            img.save(output, format=output_format, quality=85 if output_format == 'JPEG' else None)
            base64_data = base64.b64encode(output.getvalue()).decode('utf-8')
            
            return ProcessedImage(
                image_id=image_id,
                base64_data=base64_data,
                format=img_format,
                width=img.size[0],
                height=img.size[1]
            )
            
        except Exception as e:
            logger.warning(f"Failed to process image {image_id}: {e}")
            return None
    
    def render_page_as_image(self, pdf_path: str, page_num: int, dpi: int = 150) -> Optional[ProcessedImage]:
        """
        Render a PDF page as an image for vision analysis.
        
        This is useful for charts/diagrams that aren't extractable as separate images.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            dpi: Resolution for rendering
            
        Returns:
            ProcessedImage of the rendered page
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available for page rendering")
            return None
        
        try:
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                return None
            
            page = doc[page_num - 1]
            
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Resize if too large
            if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            output = io.BytesIO()
            img.save(output, format='PNG')
            base64_data = base64.b64encode(output.getvalue()).decode('utf-8')
            
            doc.close()
            
            return ProcessedImage(
                image_id=f"page_{page_num}",
                base64_data=base64_data,
                format='png',
                width=img.size[0],
                height=img.size[1],
                page=page_num
            )
            
        except Exception as e:
            logger.warning(f"Error rendering PDF page {page_num}: {e}")
            return None
    
    def select_relevant_images(
        self,
        images: List[ProcessedImage],
        criterion_text: str,
        max_images: int = 3
    ) -> List[ProcessedImage]:
        """
        Select images most relevant to a specific criterion.
        
        Args:
            images: All processed images
            criterion_text: The criterion text to match against
            max_images: Maximum number of images to return
            
        Returns:
            List of most relevant images
        """
        if not images:
            return []
        
        # Simple relevance scoring based on caption/context overlap
        criterion_words = set(criterion_text.lower().split())
        
        scored = []
        for img in images:
            score = 0
            
            # Check caption overlap
            if img.caption:
                caption_words = set(img.caption.lower().split())
                score += len(criterion_words & caption_words) * 2
            
            # Check context overlap
            if img.context:
                context_words = set(img.context.lower().split())
                score += len(criterion_words & context_words)
            
            scored.append((score, img))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return images with score > 0, up to max_images
        return [img for score, img in scored[:max_images] if score > 0]
