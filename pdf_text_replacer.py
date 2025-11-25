"""
PDF Text Replacer with Style Preservation
Replaces text in PDF while maintaining the original style (font, size, color, etc.)
"""

import fitz  # PyMuPDF
import re
from typing import List, Tuple, Optional, Dict
import io


class TextStyle:
    """Represents text style information"""
    def __init__(self, font: str = "helv", fontsize: float = 12, 
                 color: Tuple[float, float, float] = (0, 0, 0), 
                 flags: int = 0):
        self.font = font
        self.fontsize = fontsize
        self.color = color  # RGB tuple (0-1 range)
        self.flags = flags  # Text flags (bold, italic, etc.)
    
    def __repr__(self):
        return f"TextStyle(font={self.font}, size={self.fontsize}, color={self.color}, flags={self.flags})"


class PDFTextReplacer:
    """Replace text in PDF while preserving original style"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF text replacer
        
        Args:
            pdf_path: Path to the original PDF file
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    def find_text_instances(self, search_text: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Find all instances of text in the PDF with their positions and styles
        
        Args:
            search_text: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of dictionaries containing text instances with position and style info
        """
        instances = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text_dict = page.get_text("dict")
            
            # Get all text spans with their styles
            blocks = text_dict.get("blocks", [])
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_text = span["text"]
                        
                        # Check if search text is in this span
                        search_in_span = search_text if case_sensitive else search_text.lower()
                        text_to_check = span_text if case_sensitive else span_text.lower()
                        
                        if search_in_span in text_to_check:
                            # Get style information
                            style = TextStyle(
                                font=span.get("font", "helv"),
                                fontsize=span.get("size", 12),
                                color=span.get("color", 0),
                                flags=span.get("flags", 0)
                            )
                            
                            # Convert color from integer to RGB tuple
                            if isinstance(style.color, int):
                                color_int = style.color
                                r = ((color_int >> 16) & 0xFF) / 255.0
                                g = ((color_int >> 8) & 0xFF) / 255.0
                                b = (color_int & 0xFF) / 255.0
                                style.color = (r, g, b)
                            
                            # Find exact position
                            bbox = span["bbox"]  # [x0, y0, x1, y1]
                            
                            # Find character positions for the search text
                            text_start = text_to_check.find(search_in_span)
                            if text_start != -1:
                                # Calculate position of the search text
                                char_bboxes = page.get_text("rawdict")["blocks"]
                                
                                instances.append({
                                    "page": page_num,
                                    "text": span_text,
                                    "search_text": search_text,
                                    "start_pos": text_start,
                                    "bbox": bbox,
                                    "style": style,
                                    "span": span
                                })
        
        return instances
    
    def replace_text_with_style(self, old_text: str, new_text: str, 
                                case_sensitive: bool = False,
                                max_replacements: Optional[int] = None) -> int:
        """
        Replace text in PDF while preserving the original style
        
        Args:
            old_text: Text to find and replace
            new_text: Replacement text (without style - will inherit old text style)
            case_sensitive: Whether search should be case sensitive
            max_replacements: Maximum number of replacements (None = all)
        
        Returns:
            Number of replacements made
        """
        replacements = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Get all text instances
            text_instances = page.search_for(
                old_text, 
                flags=fitz.TEXT_DEHYPHENATE if not case_sensitive else 0
            )
            
            if not text_instances:
                continue
            
            # Get text blocks with style information
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])
            
            # For each found instance
            for instance_idx, rect in enumerate(text_instances):
                if max_replacements and replacements >= max_replacements:
                    break
                
                # Find the text span that contains this instance
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_bbox = span["bbox"]
                            span_text = span["text"]
                            
                            # Check if this span overlaps with our search rectangle
                            if (span_bbox[0] <= rect.x0 <= span_bbox[2] and 
                                span_bbox[1] <= rect.y0 <= span_bbox[3]):
                                
                                # Get style from this span
                                font_name = span.get("font", "helv")
                                font_size = span.get("size", 12)
                                color = span.get("color", 0)
                                flags = span.get("flags", 0)
                                
                                # Convert color from integer to RGB tuple
                                if isinstance(color, int):
                                    color_int = color
                                    r = ((color_int >> 16) & 0xFF) / 255.0
                                    g = ((color_int >> 8) & 0xFF) / 255.0
                                    b = (color_int & 0xFF) / 255.0
                                    color_rgb = (r, g, b)
                                else:
                                    color_rgb = color
                                
                                # Delete the old text (redact it)
                                page.add_redact_annot(rect)
                                page.apply_redactions()
                                
                                # Insert new text with the same style
                                point = fitz.Point(rect.x0, rect.y1)  # Position for new text
                                
                                # Set font and size
                                font_ref = fitz.Font(fontname=font_name)
                                page.insert_text(
                                    point,
                                    new_text,
                                    fontsize=font_size,
                                    fontname=font_name,
                                    color=color_rgb,
                                    render_mode=0  # Normal text
                                )
                                
                                replacements += 1
                                break
                        
                        if replacements and (max_replacements and replacements >= max_replacements):
                            break
                    
                    if replacements and (max_replacements and replacements >= max_replacements):
                        break
                
                if max_replacements and replacements >= max_replacements:
                    break
        
        return replacements
    
    def replace_text_advanced(self, old_text: str, new_text: str,
                             case_sensitive: bool = False,
                             preserve_formatting: bool = True) -> int:
        """
        Advanced text replacement with better style preservation
        
        Args:
            old_text: Text to find and replace
            new_text: Replacement text
            case_sensitive: Whether search should be case sensitive
            preserve_formatting: Whether to preserve original formatting
        
        Returns:
            Number of replacements made
        """
        replacements = 0
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Search for text
            search_flags = 0 if case_sensitive else fitz.TEXT_DEHYPHENATE
            text_rects = page.search_for(old_text, flags=search_flags)
            
            if not text_rects:
                continue
            
            # Get text with detailed formatting
            text_dict = page.get_text("dict")
            
            # Process replacements in reverse order to avoid position shifts
            for rect in reversed(text_rects):
                # Find the text span that contains this rectangle
                span_info = self._find_span_at_position(page, text_dict, rect)
                
                if span_info:
                    # Extract style from the span
                    font = span_info.get("font", "helv")
                    size = span_info.get("size", 12)
                    color = span_info.get("color", 0)
                    
                    # Convert color
                    if isinstance(color, int):
                        color_rgb = self._int_to_rgb(color)
                    else:
                        color_rgb = color
                    
                    # Delete old text using redaction
                    try:
                        page.add_redact_annot(rect)
                        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                    except Exception as e:
                        print(f"Warning: Redaction failed: {e}")
                    
                    # Calculate insertion point (align with baseline of original text)
                    # Use the bottom-left of the rect, adjusted for font size
                    insert_point = fitz.Point(rect.x0, rect.y1)
                    
                    # Insert new text with preserved style
                    try:
                        # Try to use the exact font
                        try:
                            font_ref = fitz.Font(fontname=font)
                        except:
                            # Fallback to default font if font not found
                            font_ref = fitz.Font("helv")
                            font = "helv"
                        
                        page.insert_text(
                            insert_point,
                            new_text,
                            fontsize=size,
                            fontname=font,
                            color=color_rgb,
                            render_mode=0
                        )
                        replacements += 1
                    except Exception as e:
                        print(f"Warning: Could not insert text: {e}")
                        # Try with default font as fallback
                        try:
                            page.insert_text(
                                insert_point,
                                new_text,
                                fontsize=size,
                                color=color_rgb,
                                render_mode=0
                            )
                            replacements += 1
                        except Exception as e2:
                            print(f"Error: Failed to insert text even with fallback: {e2}")
                            continue
                else:
                    print(f"Warning: Could not find span information for text at {rect}")
            
            # Update text_dict for next iteration (needed if multiple replacements)
            text_dict = page.get_text("dict")
        
        return replacements
    
    def _find_span_at_position(self, page, text_dict: Dict, rect: fitz.Rect) -> Optional[Dict]:
        """Find the text span at a given position"""
        blocks = text_dict.get("blocks", [])
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                for span in line["spans"]:
                    span_bbox = span["bbox"]
                    # Check if span overlaps with rect
                    if (span_bbox[0] <= rect.x0 <= span_bbox[2] and
                        span_bbox[1] <= rect.y0 <= span_bbox[3]):
                        return span
        
        return None
    
    def _int_to_rgb(self, color_int: int) -> Tuple[float, float, float]:
        """Convert integer color to RGB tuple (0-1 range)"""
        r = ((color_int >> 16) & 0xFF) / 255.0
        g = ((color_int >> 8) & 0xFF) / 255.0
        b = (color_int & 0xFF) / 255.0
        return (r, g, b)
    
    def save(self, output_path: str):
        """
        Save the modified PDF
        
        Args:
            output_path: Path where to save the modified PDF
        """
        self.doc.save(output_path)
    
    def save_to_bytes(self) -> bytes:
        """
        Save the modified PDF to bytes
        
        Returns:
            PDF as bytes
        """
        return self.doc.tobytes()
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()


def replace_text_in_pdf(pdf_path: str, old_text: str, new_text: str,
                        output_path: Optional[str] = None,
                        case_sensitive: bool = False) -> bytes:
    """
    Convenience function to replace text in PDF with style preservation
    
    Args:
        pdf_path: Path to input PDF file
        old_text: Text to find and replace (e.g., "xxxx")
        new_text: Replacement text (e.g., "yyyy") - will inherit old text style
        output_path: Optional path to save output PDF (if None, returns bytes)
        case_sensitive: Whether search should be case sensitive
    
    Returns:
        PDF bytes if output_path is None, otherwise saves to file
    
    Example:
        # Replace "job description : xxxx" with "job description : yyyy"
        pdf_bytes = replace_text_in_pdf(
            "resume.pdf", 
            old_text="xxxx",
            new_text="yyyy"
        )
    """
    replacer = PDFTextReplacer(pdf_path)
    
    try:
        # Replace text
        count = replacer.replace_text_advanced(old_text, new_text, case_sensitive)
        print(f"Replaced {count} instance(s) of '{old_text}' with '{new_text}'")
        
        # Save or return bytes
        if output_path:
            replacer.save(output_path)
            return None
        else:
            pdf_bytes = replacer.save_to_bytes()
            return pdf_bytes
    
    finally:
        replacer.close()


# Example usage
if __name__ == "__main__":
    # Example 1: Replace text and save to file
    # replace_text_in_pdf(
    #     "resume.pdf",
    #     old_text="xxxx",
    #     new_text="yyyy",
    #     output_path="resume_updated.pdf"
    # )
    
    # Example 2: Replace text and get bytes
    # pdf_bytes = replace_text_in_pdf(
    #     "resume.pdf",
    #     old_text="job description : xxxx",
    #     new_text="job description : yyyy"
    # )
    
    # Example 3: Advanced usage with more control
    # replacer = PDFTextReplacer("resume.pdf")
    # count = replacer.replace_text_advanced("xxxx", "yyyy")
    # replacer.save("resume_updated.pdf")
    # replacer.close()
    
    print("PDF Text Replacer module loaded. Use replace_text_in_pdf() function.")

