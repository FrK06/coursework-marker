"""
Image Extractor - Handles extraction and processing of figures from documents.

Supports:
- Extracting images from DOCX files
- Resizing for optimal model input
- Base64 encoding for API calls
- Linking images to their captions/context
"""
import io
import base64
from pathlib import Path                                                                 #  "Path" is not accessed
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an extracted image with metadata."""
    image_id: str
    image_bytes: bytes
    format: str  # 'png', 'jpeg', etc.
    width: int
    height: int
    caption: Optional[str] = None
    context: Optional[str] = None  # Text around the figure reference
    page_estimate: int = 1
    
    def to_base64(self) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(self.image_bytes).decode('utf-8')
    
    def get_data_uri(self) -> str:
        """Get data URI for embedding in HTML or API calls."""
        mime_type = f"image/{self.format}"
        return f"data:{mime_type};base64,{self.to_base64()}"


class ImageExtractor:
    """
    Extracts and processes images from documents for vision model analysis.
    
    Features:
    - Extract images from DOCX files
    - Resize large images to optimize for model input
    - Link images to their captions and surrounding context
    - Prepare images for Ollama vision API calls
    """
    
    SUPPORTED_FORMATS = {'png', 'jpeg', 'jpg', 'gif', 'webp', 'bmp'}
    
    def __init__(self, max_size: Tuple[int, int] = (800, 800)):
        """
        Initialize the image extractor.
        
        Args:
            max_size: Maximum (width, height) for resized images
        """
        if Image is None:
            raise ImportError("Pillow is required. Install with: pip install Pillow")
        
        self.max_size = max_size
    
    def process_images(
        self,
        image_dict: Dict[str, bytes],
        captions: Dict[str, str],
        chunks: List[Any]
    ) -> Dict[str, ExtractedImage]:
        """
        Process extracted images, resize them, and link to context.
        
        Args:
            image_dict: Dict of image_id -> raw image bytes
            captions: Dict of image_id -> caption text
            chunks: Document chunks to find context around figure references
            
        Returns:
            Dict of image_id -> ExtractedImage with all metadata
        """
        processed_images = {}
        
        for image_id, raw_bytes in image_dict.items():
            try:
                # Process the image
                processed = self._process_single_image(
                    image_id,
                    raw_bytes,
                    captions.get(image_id)
                )
                
                if processed:
                    # Find context from chunks
                    context = self._find_image_context(image_id, chunks)
                    processed.context = context
                    
                    # Estimate page from chunks
                    page = self._find_image_page(image_id, chunks)
                    processed.page_estimate = page
                    
                    processed_images[image_id] = processed
                    
            except Exception as e:
                logger.warning(f"Could not process image {image_id}: {e}")
        
        return processed_images
    
    def _process_single_image(
        self,
        image_id: str,
        raw_bytes: bytes,
        caption: Optional[str]
    ) -> Optional[ExtractedImage]:
        """Process a single image: validate, resize, and package."""
        try:
            # Open image
            img = Image.open(io.BytesIO(raw_bytes))
            
            # Get format
            img_format = img.format.lower() if img.format else 'png'
            if img_format == 'jpeg':
                img_format = 'jpeg'
            elif img_format not in self.SUPPORTED_FORMATS:
                # Convert to PNG if format not supported
                img_format = 'png'
            
            original_size = img.size
            
            # Resize if necessary
            if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (for JPEG)
            if img_format == 'jpeg' and img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save to bytes
            output = io.BytesIO()
            img.save(output, format=img_format.upper())
            processed_bytes = output.getvalue()
            
            return ExtractedImage(
                image_id=image_id,
                image_bytes=processed_bytes,
                format=img_format,
                width=img.size[0],
                height=img.size[1],
                caption=caption
            )
            
        except Exception as e:
            logger.warning(f"Failed to process image {image_id}: {e}")
            return None
    
    def _find_image_context(
        self,
        image_id: str,
        chunks: List[Any]
    ) -> Optional[str]:
        """
        Find the context (surrounding text) for an image reference.
        
        Looks for chunks that reference this figure and returns nearby text.
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Check if this chunk references the figure
            if hasattr(chunk, 'figure_ids') and image_id in chunk.figure_ids:
                # Add this chunk's content
                context_parts.append(chunk.content)
                
                # Add surrounding chunks for context
                if i > 0:
                    prev_chunk = chunks[i-1]
                    if hasattr(prev_chunk, 'content'):
                        context_parts.insert(0, prev_chunk.content[:200])
                
                if i < len(chunks) - 1:
                    next_chunk = chunks[i+1]
                    if hasattr(next_chunk, 'content'):
                        context_parts.append(next_chunk.content[:200])
        
        if context_parts:
            return " ... ".join(context_parts)
        
        return None
    
    def _find_image_page(
        self,
        image_id: str,
        chunks: List[Any]
    ) -> int:
        """Find the estimated page number for an image."""
        for chunk in chunks:
            if hasattr(chunk, 'figure_ids') and image_id in chunk.figure_ids:
                if hasattr(chunk, 'page_estimate'):
                    return chunk.page_estimate
                elif hasattr(chunk, 'page_number'):
                    return chunk.page_number
        return 1
    
    def prepare_for_vision(
        self,
        image: ExtractedImage,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare an image for vision model API call.
        
        Returns a dict suitable for Ollama's vision API format.
        """
        prompt_parts = []
        
        if image.caption:
            prompt_parts.append(f"Caption: {image.caption}")
        
        if include_context and image.context:
            prompt_parts.append(f"Context from document: {image.context[:500]}")
        
        return {
            'image_base64': image.to_base64(),
            'image_format': image.format,
            'context_prompt': "\n".join(prompt_parts) if prompt_parts else None,
            'image_id': image.image_id,
            'page': image.page_estimate
        }
    
    def select_relevant_images(
        self,
        images: Dict[str, ExtractedImage],
        criterion_text: str,
        max_images: int = 3
    ) -> List[ExtractedImage]:
        """
        Select images most relevant to a specific criterion.
        
        Args:
            images: All extracted images
            criterion_text: The criterion text to match against
            max_images: Maximum number of images to return
            
        Returns:
            List of most relevant images
        """
        # Simple relevance scoring based on caption/context overlap
        scored_images = []
        
        criterion_words = set(criterion_text.lower().split())
        
        for image_id, image in images.items():                                                            # "image_id" is not accessed
            score = 0
            
            # Check caption overlap
            if image.caption:
                caption_words = set(image.caption.lower().split())
                score += len(criterion_words & caption_words) * 2
            
            # Check context overlap
            if image.context:
                context_words = set(image.context.lower().split())
                score += len(criterion_words & context_words)
            
            scored_images.append((score, image))
        
        # Sort by score and return top images
        scored_images.sort(key=lambda x: x[0], reverse=True)
        
        return [img for score, img in scored_images[:max_images] if score > 0]
