"""
PDF Style Generator - Creates PDFs matching the style of a sample PDF
Uses the sample PDF's styling (fonts, colors, margins, layout) and applies it to new text content
"""

import io
import json
import os
import traceback
from typing import Dict, Optional, Tuple
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, blue
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
import pdfplumber
import re


class PDFStyleExtractor:
    """Extract style information from a sample PDF"""
    
    def __init__(self, sample_pdf_path: str):
        self.sample_pdf_path = sample_pdf_path
        self.style_info = {}
        
    def extract_styles(self) -> Dict:
        """Extract style information from the sample PDF"""
        style_info = {
            'page_size': letter,  # Default
            'margins': {'top': 1*inch, 'bottom': 1*inch, 'left': 1*inch, 'right': 1*inch},
            'font_name': 'Helvetica',  # Default
            'font_size': 12,
            'line_spacing': 14,
            'text_color': black,
            'alignment': TA_LEFT,
            'bold': False,
            'italic': False,
        }
        
        try:
            # Try to extract style info using pdfplumber
            with pdfplumber.open(self.sample_pdf_path) as pdf:
                if len(pdf.pages) > 0:
                    first_page = pdf.pages[0]
                    
                    # Extract text with font information
                    chars = first_page.chars
                    if chars:
                        # Analyze font properties from the first page
                        fonts = {}
                        font_sizes = []
                        colors = []
                        
                        for char in chars[:100]:  # Sample first 100 chars
                            font_name = char.get('fontname', 'Helvetica')
                            font_size = char.get('size', 12)
                            
                            # Extract base font name (remove subset prefixes)
                            base_font = re.sub(r'^[A-Z]{6}\+', '', font_name)
                            base_font = re.sub(r'^[A-Z]{1,2}', '', base_font)
                            
                            fonts[base_font] = fonts.get(base_font, 0) + 1
                            font_sizes.append(font_size)
                            
                            # Extract color if available
                            if 'ncolor' in char and char['ncolor']:
                                colors.append(tuple(char['ncolor']))
                        
                        # Get most common font
                        if fonts:
                            most_common_font = max(fonts.items(), key=lambda x: x[1])[0]
                            style_info['font_name'] = most_common_font
                        
                        # Get average font size
                        if font_sizes:
                            style_info['font_size'] = sum(font_sizes) / len(font_sizes)
                        
                        # Get most common color
                        if colors:
                            from collections import Counter
                            color_counts = Counter(colors)
                            most_common_color = color_counts.most_common(1)[0][0]
                            if most_common_color != (0, 0, 0):  # Not black
                                style_info['text_color'] = most_common_color
                    
                    # Extract page dimensions
                    width = first_page.width
                    height = first_page.height
                    
                    # Determine page size
                    if abs(width - 612) < 10 and abs(height - 792) < 10:  # Letter size
                        style_info['page_size'] = letter
                    elif abs(width - 595) < 10 and abs(height - 842) < 10:  # A4
                        style_info['page_size'] = A4
                    
                    # Estimate margins by analyzing text positions
                    if chars:
                        x_positions = [char['x0'] for char in chars]
                        y_positions = [char['top'] for char in chars]
                        
                        if x_positions:
                            left_margin = min(x_positions)
                            right_margin = width - max([char['x1'] for char in chars])
                            style_info['margins']['left'] = left_margin
                            style_info['margins']['right'] = right_margin
                        
                        if y_positions:
                            top_margin = height - max(y_positions)
                            bottom_margin = min([char['bottom'] for char in chars])
                            style_info['margins']['top'] = top_margin
                            style_info['margins']['bottom'] = bottom_margin
                
        except Exception as e:
            print(f"Warning: Could not extract all styles from PDF: {e}")
            print("Using default styles...")
        
        self.style_info = style_info
        return style_info
    
    def save_styles(self, output_path: str) -> bool:
        """
        Save extracted styles to a JSON file
        
        Args:
            output_path: Path to save the style JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.style_info:
            self.extract_styles()
        
        try:
            # Convert style info to serializable format
            serializable_styles = self._style_to_dict(self.style_info)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_styles, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving styles: {e}")
            print("\nFull error traceback:")
            traceback.print_exc()
            return False
    
    def _style_to_dict(self, style_info: Dict) -> Dict:
        """Convert style info to JSON-serializable dictionary"""
        serializable = {}
        
        # Convert page size
        if 'page_size' in style_info:
            page_size = style_info['page_size']
            if isinstance(page_size, tuple):
                serializable['page_size'] = {'width': page_size[0], 'height': page_size[1]}
            elif page_size == letter:
                serializable['page_size'] = 'letter'
            elif page_size == A4:
                serializable['page_size'] = 'A4'
            else:
                serializable['page_size'] = 'letter'  # Default
        
        # Convert margins (already in points, convert to inches)
        if 'margins' in style_info:
            margins = style_info['margins']
            if isinstance(margins, dict):
                try:
                    serializable['margins'] = {
                        'top': float(margins.get('top', 1 * inch)) / inch if 'top' in margins else 1.0,
                        'bottom': float(margins.get('bottom', 1 * inch)) / inch if 'bottom' in margins else 1.0,
                        'left': float(margins.get('left', 1 * inch)) / inch if 'left' in margins else 1.0,
                        'right': float(margins.get('right', 1 * inch)) / inch if 'right' in margins else 1.0,
                    }
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not convert margins to serializable format: {e}")
                    # Use default margins
                    serializable['margins'] = {
                        'top': 1.0,
                        'bottom': 1.0,
                        'left': 1.0,
                        'right': 1.0,
                    }
            else:
                # If margins is not a dict, use defaults
                serializable['margins'] = {
                    'top': 1.0,
                    'bottom': 1.0,
                    'left': 1.0,
                    'right': 1.0,
                }
        
        # Font name (already string)
        if 'font_name' in style_info:
            serializable['font_name'] = style_info['font_name']
        
        # Font size (already number)
        if 'font_size' in style_info:
            serializable['font_size'] = float(style_info['font_size'])
        
        # Line spacing (already number)
        if 'line_spacing' in style_info:
            serializable['line_spacing'] = float(style_info['line_spacing'])
        
        # Convert color
        if 'text_color' in style_info:
            color = style_info['text_color']
            try:
                if isinstance(color, tuple) and len(color) >= 3:
                    # RGB tuple (0-1 range)
                    serializable['text_color'] = {
                        'type': 'rgb',
                        'r': float(color[0]),
                        'g': float(color[1]),
                        'b': float(color[2])
                    }
                elif isinstance(color, HexColor):
                    # Hex color
                    serializable['text_color'] = {
                        'type': 'hex',
                        'value': str(color)
                    }
                else:
                    # Default to black
                    serializable['text_color'] = {
                        'type': 'rgb',
                        'r': 0.0,
                        'g': 0.0,
                        'b': 0.0
                    }
            except (TypeError, ValueError, IndexError) as e:
                print(f"Warning: Could not convert color to serializable format: {e}")
                # Default to black
                serializable['text_color'] = {
                    'type': 'rgb',
                    'r': 0.0,
                    'g': 0.0,
                    'b': 0.0
                }
        
        # Convert alignment
        if 'alignment' in style_info:
            align = style_info['alignment']
            if align == TA_LEFT:
                serializable['alignment'] = 'left'
            elif align == TA_CENTER:
                serializable['alignment'] = 'center'
            elif align == TA_RIGHT:
                serializable['alignment'] = 'right'
            elif align == TA_JUSTIFY:
                serializable['alignment'] = 'justify'
            else:
                serializable['alignment'] = 'left'
        
        # Other boolean flags
        if 'bold' in style_info:
            serializable['bold'] = bool(style_info['bold'])
        if 'italic' in style_info:
            serializable['italic'] = bool(style_info['italic'])
        
        return serializable


def load_saved_styles(style_json_path: str) -> Dict:
    """
    Load saved styles from a JSON file
    
    Args:
        style_json_path: Path to the saved style JSON file
        
    Returns:
        Dictionary with style information ready for use with create_styled_pdf
    """
    try:
        with open(style_json_path, 'r', encoding='utf-8') as f:
            saved_styles = json.load(f)
        
        # Convert back to format needed for PDF creation
        style_info = _dict_to_style(saved_styles)
        return style_info
    
    except Exception as e:
        print(f"Error loading saved styles: {e}")
        raise


def _dict_to_style(saved_styles: Dict) -> Dict:
    """Convert saved JSON dictionary back to style info format"""
    style_info = {}
    
    # Convert page size
    if 'page_size' in saved_styles:
        page_size = saved_styles['page_size']
        if isinstance(page_size, dict):
            style_info['page_size'] = (page_size['width'], page_size['height'])
        elif page_size == 'letter':
            style_info['page_size'] = letter
        elif page_size == 'A4':
            style_info['page_size'] = A4
        else:
            style_info['page_size'] = letter
    else:
        style_info['page_size'] = letter
    
    # Convert margins (inches to points)
    if 'margins' in saved_styles:
        margins = saved_styles['margins']
        style_info['margins'] = {
            'top': margins.get('top', 1.0) * inch,
            'bottom': margins.get('bottom', 1.0) * inch,
            'left': margins.get('left', 1.0) * inch,
            'right': margins.get('right', 1.0) * inch,
        }
    else:
        style_info['margins'] = {
            'top': 1 * inch,
            'bottom': 1 * inch,
            'left': 1 * inch,
            'right': 1 * inch,
        }
    
    # Font name
    if 'font_name' in saved_styles:
        style_info['font_name'] = saved_styles['font_name']
    else:
        style_info['font_name'] = 'Helvetica'
    
    # Font size
    if 'font_size' in saved_styles:
        style_info['font_size'] = float(saved_styles['font_size'])
    else:
        style_info['font_size'] = 12
    
    # Line spacing
    if 'line_spacing' in saved_styles:
        style_info['line_spacing'] = float(saved_styles['line_spacing'])
    else:
        style_info['line_spacing'] = 14
    
    # Convert color
    if 'text_color' in saved_styles:
        color_info = saved_styles['text_color']
        if isinstance(color_info, dict):
            if color_info.get('type') == 'rgb':
                style_info['text_color'] = (
                    float(color_info.get('r', 0.0)),
                    float(color_info.get('g', 0.0)),
                    float(color_info.get('b', 0.0))
                )
            elif color_info.get('type') == 'hex':
                style_info['text_color'] = HexColor(color_info.get('value', '#000000'))
            else:
                style_info['text_color'] = black
        else:
            style_info['text_color'] = black
    else:
        style_info['text_color'] = black
    
    # Convert alignment
    if 'alignment' in saved_styles:
        align = saved_styles['alignment'].lower()
        if align == 'left':
            style_info['alignment'] = TA_LEFT
        elif align == 'center':
            style_info['alignment'] = TA_CENTER
        elif align == 'right':
            style_info['alignment'] = TA_RIGHT
        elif align == 'justify':
            style_info['alignment'] = TA_JUSTIFY
        else:
            style_info['alignment'] = TA_LEFT
    else:
        style_info['alignment'] = TA_LEFT
    
    # Boolean flags
    if 'bold' in saved_styles:
        style_info['bold'] = bool(saved_styles['bold'])
    if 'italic' in saved_styles:
        style_info['italic'] = bool(saved_styles['italic'])
    
    return style_info


def extract_and_save_pdf_style(sample_pdf_path: str, style_output_path: str) -> bool:
    """
    Extract styles from a sample PDF and save them to a JSON file
    
    Args:
        sample_pdf_path: Path to the sample PDF file
        style_output_path: Path where to save the style JSON file
        
    Returns:
        True if successful, False otherwise
    """
    extractor = PDFStyleExtractor(sample_pdf_path)
    extractor.extract_styles()
    return extractor.save_styles(style_output_path)


def create_styled_pdf(
    text: str,
    sample_pdf_path: Optional[str] = None,
    saved_style_path: Optional[str] = None,
    output_path: Optional[str] = None,
    custom_styles: Optional[Dict] = None
) -> bytes:
    """
    Create a PDF with text content matching the style of a sample PDF or saved style
    
    Args:
        text: The text content to put in the PDF
        sample_pdf_path: Path to the sample PDF file (optional)
        saved_style_path: Path to a saved style JSON file (optional, takes precedence over sample_pdf_path)
        output_path: Path to save the PDF (optional, if None returns bytes)
        custom_styles: Optional dictionary to override extracted styles
    
    Returns:
        PDF bytes if output_path is None, otherwise saves to file
    """
    # Load styles from saved file if provided (takes precedence)
    style_info = {}
    if saved_style_path and os.path.exists(saved_style_path):
        style_info = load_saved_styles(saved_style_path)
    # Otherwise extract styles from sample PDF if provided
    elif sample_pdf_path:
        extractor = PDFStyleExtractor(sample_pdf_path)
        style_info = extractor.extract_styles()
    
    # Override with custom styles if provided
    if custom_styles:
        style_info.update(custom_styles)
    
    # Set defaults if not extracted
    page_size = style_info.get('page_size', letter)
    margins = style_info.get('margins', {
        'top': 1*inch,
        'bottom': 1*inch,
        'left': 1*inch,
        'right': 1*inch
    })
    font_name = style_info.get('font_name', 'Helvetica')
    font_size = style_info.get('font_size', 12)
    text_color = style_info.get('text_color', black)
    alignment = style_info.get('alignment', TA_LEFT)
    line_spacing = style_info.get('line_spacing', font_size * 1.2)
    
    # Create PDF in memory
    buffer = io.BytesIO()
    
    # Create document with margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=margins['right'],
        leftMargin=margins['left'],
        topMargin=margins['top'],
        bottomMargin=margins['bottom']
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    # Create custom paragraph style matching sample PDF
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=font_size,
        textColor=text_color,
        alignment=alignment,
        leading=line_spacing,
        spaceAfter=12,
    )
    
    # Convert color if it's a tuple (RGB)
    if isinstance(text_color, tuple):
        try:
            r, g, b = text_color
            text_color = HexColor(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
            custom_style.textColor = text_color
        except:
            pass
    
    # Build PDF content
    story = []
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    if not paragraphs:
        paragraphs = text.split('\n')
    
    for para in paragraphs:
        if para.strip():
            # Escape special characters for ReportLab
            para_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(para_escaped, custom_style))
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Save to file if output_path provided
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
    
    return pdf_bytes


def create_styled_pdf_advanced(
    text: str,
    sample_pdf_path: Optional[str] = None,
    output_path: Optional[str] = None,
    font_name: Optional[str] = None,
    font_size: Optional[float] = None,
    font_color: Optional[Tuple[float, float, float]] = None,
    alignment: Optional[str] = None,
    margins: Optional[Dict[str, float]] = None,
    line_spacing: Optional[float] = None,
    page_size: Optional[Tuple[float, float]] = None
) -> bytes:
    """
    Advanced version with explicit style parameters
    
    Args:
        text: The text content
        sample_pdf_path: Path to sample PDF for style extraction
        output_path: Optional output file path
        font_name: Override font name (e.g., 'Helvetica', 'Times-Roman', 'Courier')
        font_size: Override font size in points
        font_color: Override color as RGB tuple (0-1 range) or hex string
        alignment: Override alignment ('left', 'center', 'right', 'justify')
        margins: Override margins dict with 'top', 'bottom', 'left', 'right' in inches
        line_spacing: Override line spacing
        page_size: Override page size tuple (width, height) in points
    
    Returns:
        PDF bytes
    """
    # Extract base styles from sample if provided
    base_styles = {}
    if sample_pdf_path:
        extractor = PDFStyleExtractor(sample_pdf_path)
        base_styles = extractor.extract_styles()
    
    # Build custom styles dict
    custom_styles = {}
    
    if font_name:
        custom_styles['font_name'] = font_name
    elif 'font_name' in base_styles:
        custom_styles['font_name'] = base_styles['font_name']
    
    if font_size:
        custom_styles['font_size'] = font_size
    elif 'font_size' in base_styles:
        custom_styles['font_size'] = base_styles['font_size']
    
    if font_color:
        if isinstance(font_color, str):
            # Hex color
            custom_styles['text_color'] = HexColor(font_color)
        else:
            # RGB tuple
            custom_styles['text_color'] = font_color
    elif 'text_color' in base_styles:
        custom_styles['text_color'] = base_styles['text_color']
    
    if alignment:
        align_map = {
            'left': TA_LEFT,
            'center': TA_CENTER,
            'right': TA_RIGHT,
            'justify': TA_JUSTIFY
        }
        custom_styles['alignment'] = align_map.get(alignment.lower(), TA_LEFT)
    elif 'alignment' in base_styles:
        custom_styles['alignment'] = base_styles['alignment']
    
    if margins:
        from reportlab.lib.units import inch
        custom_styles['margins'] = {
            'top': margins.get('top', 1) * inch,
            'bottom': margins.get('bottom', 1) * inch,
            'left': margins.get('left', 1) * inch,
            'right': margins.get('right', 1) * inch,
        }
    elif 'margins' in base_styles:
        custom_styles['margins'] = base_styles['margins']
    
    if line_spacing:
        custom_styles['line_spacing'] = line_spacing
    
    if page_size:
        custom_styles['page_size'] = page_size
    elif 'page_size' in base_styles:
        custom_styles['page_size'] = base_styles['page_size']
    
    return create_styled_pdf(text, custom_styles=custom_styles, output_path=output_path)


# Example usage function for Streamlit
def create_downloadable_pdf_streamlit(
    text: str,
    sample_pdf_path: Optional[str] = None,
    saved_style_path: Optional[str] = None,
    filename: str = "output.pdf"
):
    """
    Create a PDF for Streamlit download button
    
    Args:
        text: Text content for PDF
        sample_pdf_path: Path to sample PDF for style matching
        saved_style_path: Path to saved style JSON file (takes precedence)
        filename: Name for the downloaded file
    
    Returns:
        tuple: (pdf_bytes, filename) for use with st.download_button
    """
    pdf_bytes = create_styled_pdf(
        text,
        sample_pdf_path=sample_pdf_path,
        saved_style_path=saved_style_path
    )
    return pdf_bytes, filename


def list_saved_styles(directory: str = ".") -> list:
    """
    List all saved style JSON files in a directory
    
    Args:
        directory: Directory to search for style files
        
    Returns:
        List of style file paths
    """
    style_files = []
    for file in os.listdir(directory):
        if file.endswith('_style.json') or file.endswith('.style.json'):
            style_files.append(os.path.join(directory, file))
    return style_files


if __name__ == "__main__":
    # Example usage
    sample_text = """This is a sample document.
    
It has multiple paragraphs and demonstrates the styling capabilities.

You can customize fonts, colors, margins, and more!"""
    
    # Example 1: Create PDF with default styles
    # pdf_bytes = create_styled_pdf(sample_text, output_path="output_default.pdf")
    # print("Created output_default.pdf")
    
    # Example 2: Extract and save styles from a sample PDF
    try:
        extract_and_save_pdf_style("Andrew-Long-4.pdf", "sample_style.json")
        print("Saved styles to sample_style.json")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFull error traceback:")
        traceback.print_exc()
    
    # Example 3: Create PDF using saved styles
    # pdf_bytes = create_styled_pdf(
    #     sample_text,
    #     saved_style_path="sample_style.json",
    #     output_path="output_with_saved_style.pdf"
    # )
    # print("Created output_with_saved_style.pdf")
    
    # Example 4: Create PDF matching sample (if you have a sample PDF)
    # pdf_bytes = create_styled_pdf(sample_text, sample_pdf_path="sample.pdf", output_path="output_styled.pdf")
    # print("Created output_styled.pdf")

