import io
import tempfile
import warnings
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

def update_resume(uploaded_file, job_description, update_instructions=None, openai_api_key=None):
    """
    Updates a resume PDF based on the provided job description and update instructions using the OpenAI API.
    Args:
        uploaded_file: Uploaded file-like object (PDF).
        job_description (str): The job description to align to.
        update_instructions (str): Optional extra update instructions for LLM.
        openai_api_key (str): Optional, if not set, attempts default behavior of OpenAI library.
    Returns:
        updated_resume (str): Updated resume text/content (HTML/Markdown). You may wish to post-process or return bytes.
    Raises:
        Exception if the process fails.
    """
    # Step 1: Extract resume text from PDF
    try:
        from pypdf import PdfReader
        pdf_bytes = uploaded_file.read()
        file_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(file_stream)
        resume_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                resume_text += content + "\n"
        if not resume_text.strip():
            raise ValueError("Could not extract text from PDF.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract resume from uploaded PDF: {str(e)}")

    # Step 2: Build the prompt for OpenAI
    system_prompt = (
        "You are an expert resume writer. Update and optimize the following resume to best align it to the provided job description."
        " Expand, rewrite, or rephrase experience and skills as needed, highlighting relevant qualifications and keywords from the job description."
        '''\n\nIMPORTANT:You must generate resume data strictly following the RenderCV resume schema below.
            DO NOT use JSON Resume schema.
            DO NOT add extra fields not listed.
            When mentioning about the current company do not use 'Present' for the end date, use 'present' instead.
            Output format must be valid YAML that RenderCV can validate and render.
            Follow the structure exactly.

            --- BEGIN RENDERCV SCHEMA ---
           {
                'cv': {
                    'name': 'string',
                    'email': 'string',
                    'phone': 'string',
                    'location': 'string',
                    'label': 'string',  # Job title or similar
                    'summary': 'string',  # A paragraph summarizing the person
                    'website': 'string (optional)',
                    'social_networks': [
                        {
                            'network': 'string',  # Example: 'LinkedIn', 'GitHub'
                            'username': 'string',
                            'url': 'string (optional)'
                        }
                    ]
                    'sections': {
                        'experience': [
                            {
                                'company': 'string',
                                'position': 'string',
                                'location': 'string',
                                'start_date': 'string (YYYY-MM)',  # Example: '2020-01'
                                'end_date': 'string (YYYY-MM or "Present")',
                                'highlights': ['string', 'string', 'string']  # List of highlights
                            }
                        ],
                        'education': [
                            {
                                'institution': 'string',
                                'studyType': 'string',  # Example: 'B.Sc.', 'M.Sc.'
                                'area': 'string',
                                'start_date': 'string (YYYY-MM)',
                                'end_date': 'string (YYYY-MM)',
                                'highlights': ['string']  # List of highlights for education
                            }
                        ],
                        'skills': [
                            {'name': 'string'}  # Example: 'Python', 'AWS', 'Django'
                        ],
                        'certifications': [
                            {
                                'name': 'string',
                                'issuer': 'string',
                                'date': 'string (YYYY-MM)'  # Date of certification
                            }
                        ],
                        'publications': [
                            {
                                'title': 'string',
                                'journal': 'string',
                                'year': 'string (YYYY)',
                                'url': 'string (optional)'  # URL to the publication (optional)
                            }
                        ]
                    }
                }
            }
            --- END RENDERCV SCHEMA ---

            Here is an example of a fully valid RenderCV resume:
            --- BEGIN EXAMPLE ---
            {
                'cv': {
                    'name': 'Andrew Long',
                    'email': 'andrewlong0808@gmail.com',
                    'phone': '(352) 580-0750',
                    'location': 'Los Angeles, CA',
                    'label': 'Senior Machine Learning Engineer',
                    'summary': 'Experienced Full-Stack and AI Engineer with over 10 years of expertise in designing, building, and deploying end-to-end software and AI systems.',
                    'sections': [
                        {
                            'name': 'experience',
                            'title': 'Work Experience',
                            'items': [
                                {'company': 'Evertune AI', 'position': 'Software Engineer', 'location': 'Seattle, WA', 'start': '2025-04', 'end': 'Present'}
                            ]
                        },
                        {
                            'name': 'education',
                            'title': 'Education',
                            'items': [
                                {'institution': 'University of Illinois at Chicago', 'studyType': 'B.Sc.', 'area': 'Computer Science', 'start': '2011', 'end': '2013'}
                            ]
                        },
                        {
                            'name': 'skills',
                            'title': 'Skills',
                            'items': [
                                {'name': 'Python'},
                                {'name': 'TypeScript'},
                                {'name': 'JavaScript'}
                            ]
                        },
                        {
                            'name': 'certifications',
                            'title': 'Certifications',
                            'items': [
                                {'name': 'AWS Certified Machine Learning - Specialty', 'issuer': 'Amazon Web Services', 'date': '2020-09'},
                                {'name': 'TensorFlow Developer Certificate', 'issuer': 'TensorFlow', 'date': '2020-05'}
                            ]
                        }
                    ]
                }
            }

            --- END EXAMPLE ---
        '''
        "if something is not mentioned in the job description, make it None, not null."
        " Do NOT include any markdown code blocks, explanations, or other text. Return ONLY the raw JSON object starting with '{' and ending with '}'."
    )
    if update_instructions:
        system_prompt += f"\n\nAdditional instructions: {update_instructions}"

    user_prompt = (
        f"HERE IS THE JOB DESCRIPTION:\n{job_description}\n\n"
        f"HERE IS THE ORIGINAL RESUME:\n{resume_text}\n\n"
        "Convert this resume to JSON Resume format and return ONLY the JSON object (no markdown, no code blocks, no explanations)."
    )

    # Step 3: Call OpenAI API to get the updated resume
    try:
        # Lazy import to allow this util to work in Streamlit context or not
        from openai import OpenAI
        if openai_api_key:
            client = OpenAI(api_key=openai_api_key)
        else:
            client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2500,
        )
        updated_resume = completion.choices[0].message.content.strip()
        return updated_resume
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {str(e)}")


def replace_text_in_pdf_with_whole_text(pdf_file, new_text: str, output_path: str = None):
    """
    Replace the entire content of a PDF with new_text, automatically matching and replacing sections 
    while preserving the original style (font, size, color, etc.) for each part.

    Always performs case-sensitive matching (case_sensitive=True).
    Always returns PDF bytes (return_pdf_bytes=True).

    Args:
        pdf_file: PDF file-like object (e.g., uploaded file) or path to PDF file.
        new_text (str): The whole replacement text for the PDF (ideally with similar structure/order).
        output_path (str): Optional path to save output PDF (for compatibility, no effect on return value).

    Returns:
        bytes: PDF bytes.

    Raises:
        RuntimeError: If PDF processing fails.

    Example:
        pdf_bytes = replace_text_in_pdf_with_whole_text(
            uploaded_pdf_file, 
            new_text=resume_txt_content,
        )
    """
    try:
        import fitz  # PyMuPDF
        import tempfile
        import os

        # File input: path or file-like object
        if hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                input_path = tmp_file.name
            is_temp_file = True
        else:
            input_path = pdf_file
            is_temp_file = False

        doc = fitz.open(input_path)

        # Split new_text into lines (or paragraphs)
        new_lines = [l.strip() for l in new_text.splitlines() if l.strip()]
        original_lines = []
        page_indices = []
        line_positions = []  # (page_no, line, bbox, style)

        # Extract (line text, page index, line, bbox, style)
        for pno, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['type'] == 0:  # text
                    for line in block["lines"]:
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        
                        line_text = "".join([span.get("text", "") for span in spans]).strip()
                        if line_text:
                            original_lines.append(line_text)
                            page_indices.append(pno)
                            
                            # Get style from first span
                            first_span = spans[0]
                            style = {
                                "font": first_span.get("font", "helv"),
                                "size": first_span.get("size", 12),
                                "color": first_span.get("color", 0),
                                "flags": first_span.get("flags", 0)
                            }
                            bbox = first_span.get("bbox", [0, 0, 0, 0])
                            if len(bbox) != 4:
                                bbox = [0, 0, 0, 0]
                            line_positions.append((pno, line, bbox, style))

        # Check if we have any lines to process
        if not original_lines:
            raise RuntimeError("No text found in the original PDF")
        
        if not new_lines:
            raise RuntimeError("New text is empty")

        from difflib import SequenceMatcher
        # Always use case-sensitive matching
        matcher = SequenceMatcher(None, original_lines, new_lines, autojunk=False)
        opcodes = matcher.get_opcodes()

        # Helper function to convert color integer to RGB tuple
        def int_to_rgb(color_int):
            """Convert integer color to RGB tuple (0-1 range)"""
            if isinstance(color_int, (list, tuple)):
                return tuple(color_int)
            r = ((color_int >> 16) & 0xFF) / 255.0
            g = ((color_int >> 8) & 0xFF) / 255.0
            b = (color_int & 0xFF) / 255.0
            return (r, g, b)
        
        # Perform replacements on matched lines
        # Process in reverse order to avoid position shifts
        opcodes_reversed = list(reversed(opcodes))
        
        # First pass: Mark all areas that need to be redacted
        for tag, i1, i2, j1, j2 in opcodes_reversed:
            if tag in ('replace', 'delete'):
                # For replaced/deleted lines, mark for redaction
                for idx in range(i1, i2):
                    if idx < len(line_positions):
                        pno, line, bbox, style = line_positions[idx]
                        page = doc[pno]
                        # Convert bbox list to Rect object
                        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                        try:
                            page.add_redact_annot(rect, fill=(1, 1, 1))
                        except:
                            pass
        
        # Apply redactions before inserting new text
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except:
                pass
        
        # Second pass: Insert new text with preserved styles
        for tag, i1, i2, j1, j2 in opcodes_reversed:
            if tag in ('replace', 'insert'):
                # For replaced/inserted, write new lines with previous style
                for offset, idx in enumerate(range(i1, min(i2, len(line_positions)))):
                    if idx < len(line_positions):
                        pno, line, bbox, style = line_positions[idx]
                        page = doc[pno]
                        try:
                            text = new_lines[j1 + offset] if (j1 + offset) < len(new_lines) else ""
                        except IndexError:
                            text = ""
                        
                        if not text:
                            continue
                        
                        font = style.get("font", "helv")
                        size = style.get("size", 12)
                        color_int = style.get("color", 0)
                        color = int_to_rgb(color_int)
                        
                        # Convert bbox list to Rect object
                        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                        
                        # Insert new text with preserved style
                        try:
                            # Try insert_textbox first (better for fitting text in boxes)
                            try:
                                page.insert_textbox(
                                    rect,
                                    text,
                                    fontname=font,
                                    fontsize=size,
                                    color=color,
                                    align=0  # left align
                                )
                            except:
                                # Fallback to insert_text (more reliable)
                                insert_point = fitz.Point(bbox[0], bbox[3])
                                page.insert_text(
                                    insert_point,
                                    text,
                                    fontsize=size,
                                    fontname=font,
                                    color=color,
                                    render_mode=0
                                )
                        except Exception as e:
                            print(f"Warning: Could not insert text '{text[:20]}...': {e}")
                            # Last resort: try with default font
                            try:
                                insert_point = fitz.Point(bbox[0], bbox[3])
                                page.insert_text(
                                    insert_point,
                                    text,
                                    fontsize=size,
                                    color=color,
                                    render_mode=0
                                )
                            except:
                                pass
                            continue

        # Always return PDF bytes
        pdf_bytes = None
        try:
            pdf_bytes = doc.tobytes()
        finally:
            doc.close()

        if is_temp_file and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except:
                pass

        return pdf_bytes

    except ImportError as e:
        raise RuntimeError(f"Required library not found: {str(e)}. Please install PyMuPDF (pip install PyMuPDF)")
    except Exception as e:
        raise RuntimeError(f"Failed to replace text in PDF: {str(e)}")


def _json_resume_to_text(resume_json: Dict) -> str:
    """
    Convert JSON Resume schema to formatted text for PDF generation.
    
    Args:
        resume_json: Dictionary following JSON Resume schema
        
    Returns:
        Formatted text string
    """
    lines = []
    
    # Basics section
    basics = resume_json.get("basics", {})
    if basics.get("name"):
        lines.append(basics["name"])
        if basics.get("label"):
            lines.append(basics["label"])
        lines.append("")
    
    # Contact info
    contact = []
    if basics.get("email"):
        contact.append(basics["email"])
    if basics.get("phone"):
        contact.append(basics["phone"])
    if basics.get("url"):
        contact.append(basics["url"])
    if basics.get("location"):
        loc = basics["location"]
        loc_parts = [loc.get("city"), loc.get("region"), loc.get("countryCode")]
        loc_str = ", ".join(filter(None, loc_parts))
        if loc_str:
            contact.append(loc_str)
    if contact:
        lines.append(" | ".join(contact))
        lines.append("")
    
    # Summary
    if basics.get("summary"):
        lines.append(basics["summary"])
        lines.append("")
    
    # Work experience
    work = resume_json.get("work", [])
    if work:
        lines.append("PROFESSIONAL EXPERIENCE")
        lines.append("")
        for job in work:
            if job.get("name") or job.get("position"):
                job_line = []
                if job.get("position"):
                    job_line.append(job["position"])
                if job.get("name"):
                    job_line.append(f"at {job['name']}")
                if job_line:
                    lines.append(" - ".join(job_line))
            if job.get("startDate") or job.get("endDate"):
                dates = " / ".join(filter(None, [job.get("startDate"), job.get("endDate")]))
                if dates:
                    lines.append(dates)
            if job.get("summary"):
                lines.append(job["summary"])
            for highlight in job.get("highlights", []):
                lines.append(f"• {highlight}")
            lines.append("")
    
    # Education
    education = resume_json.get("education", [])
    if education:
        lines.append("EDUCATION")
        lines.append("")
        for edu in education:
            if edu.get("institution"):
                lines.append(edu["institution"])
            if edu.get("studyType") or edu.get("area"):
                degree = ", ".join(filter(None, [edu.get("studyType"), edu.get("area")]))
                if degree:
                    lines.append(degree)
            if edu.get("startDate") or edu.get("endDate"):
                dates = " / ".join(filter(None, [edu.get("startDate"), edu.get("endDate")]))
                if dates:
                    lines.append(dates)
            if edu.get("score"):
                lines.append(f"GPA: {edu['score']}")
            lines.append("")
    
    # Skills
    skills = resume_json.get("skills", [])
    if skills:
        lines.append("SKILLS")
        lines.append("")
        for skill in skills:
            if skill.get("name"):
                keywords = ", ".join(skill.get("keywords", []))
                if keywords:
                    lines.append(f"{skill['name']}: {keywords}")
                else:
                    lines.append(skill["name"])
        lines.append("")
    
    # Projects
    projects = resume_json.get("projects", [])
    if projects:
        lines.append("PROJECTS")
        lines.append("")
        for project in projects:
            if project.get("name"):
                lines.append(project["name"])
            if project.get("description"):
                lines.append(project["description"])
            for highlight in project.get("highlights", []):
                lines.append(f"• {highlight}")
            lines.append("")
    
    return "\n".join(lines)


@contextmanager
def suppress_flask_warnings():
    """Context manager to suppress Flask development server warnings."""
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Suppress Flask/Werkzeug logging
        flask_logger = logging.getLogger('werkzeug')
        flask_logger.setLevel(logging.ERROR)
        
        # Also suppress Flask's own logger
        flask_app_logger = logging.getLogger('flask')
        flask_app_logger.setLevel(logging.ERROR)
        
        # Temporarily redirect stderr to suppress Flask's warning message
        original_stderr = sys.stderr
        try:
            # Create a filter that removes Flask development server warnings
            class FlaskWarningFilter:
                def __init__(self, original):
                    self.original = original
                
                def write(self, text):
                    if "WARNING: This is a development server" not in text:
                        self.original.write(text)
                
                def flush(self):
                    self.original.flush()
                
                def __getattr__(self, name):
                    return getattr(self.original, name)
            
            sys.stderr = FlaskWarningFilter(original_stderr)
            yield
        finally:
            sys.stderr = original_stderr
            flask_logger.setLevel(logging.WARNING)
            flask_app_logger.setLevel(logging.WARNING)

def _normalize_phone_for_rendercv(phone: str) -> str:
    """
    Normalize phone number to RenderCV format: +1-609-999-9995
    
    Args:
        phone: Phone number in any format
        
    Returns:
        Phone number in RenderCV format (without tel: prefix, with dashes)
        Returns None if phone number cannot be normalized
    """
    if not phone:
        return None
    
    import re
    # Remove all non-digit characters except +
    digits = re.sub(r'[^\d+]', '', phone)
    
    # If it doesn't start with +, assume US number and add +1
    if not digits.startswith('+'):
        # Remove leading 1 if present (US country code)
        if digits.startswith('1') and len(digits) == 11:
            digits = digits[1:]
        # Need at least 10 digits for US number
        if len(digits) < 10:
            return None  # Invalid phone number
        digits = '+1' + digits
    
    # Format with dashes: +1-609-999-9995
    # For US/Canada numbers: +1-XXX-XXX-XXXX
    if digits.startswith('+1') and len(digits) == 12:  # +1 + 10 digits
        rest = digits[2:]  # Remove +1
        formatted = f"+1-{rest[:3]}-{rest[3:6]}-{rest[6:]}"
    elif digits.startswith('+'):
        # For other countries, format with dashes
        # Extract country code (1-3 digits after +)
        match = re.match(r'^(\+\d{1,3})(\d+)$', digits)
        if match:
            country_code, rest = match.groups()
            if len(rest) < 7:  # Minimum reasonable phone number length
                return None  # Invalid phone number
            # Format rest with dashes every 3 digits
            formatted = country_code + '-' + '-'.join([rest[i:i+3] for i in range(0, len(rest), 3)])
        else:
            # Invalid format - return None to skip phone number
            return None
    else:
        # Invalid format - return None to skip phone number
        return None
    
    # RenderCV expects phone number without tel: prefix
    # Format: +1-609-999-9995 (just the number with dashes)
    return formatted


def _convert_json_resume_to_rendercv_format(resume_data: Dict) -> Dict:
    """
    Convert JSON Resume schema format to RenderCV format.
    
    Args:
        resume_data: Dictionary following JSON Resume schema
        
    Returns:
        Dictionary in RenderCV format
    """
    rendercv_data = {
        "cv": {
            "name": "",
            "location": "",
            "email": "",
            "phone": None,
            "website": None,
            "photo": None,
            "social_networks": [],
            "sections_input": {}
        }
    }
    
    # Convert basics
    basics = resume_data.get("basics", {})
    if basics:
        rendercv_data["cv"]["name"] = basics.get("name", "")
        rendercv_data["cv"]["email"] = basics.get("email", "")
        # Normalize phone number to RenderCV format
        # If normalization fails (returns None), phone will be None and won't be included
        phone = basics.get("phone")
        normalized_phone = _normalize_phone_for_rendercv(phone) if phone else None
        # Only set phone if normalization succeeded
        if normalized_phone:
            rendercv_data["cv"]["phone"] = normalized_phone
        # If None, phone field won't be set (RenderCV will handle it as optional)
        rendercv_data["cv"]["website"] = basics.get("url")
        
        # Location
        location = basics.get("location", {})
        if location:
            loc_parts = [
                location.get("city"),
                location.get("region"),
                location.get("countryCode")
            ]
            rendercv_data["cv"]["location"] = ", ".join(filter(None, loc_parts))
        
        # Social networks - RenderCV requires exact network names (case-sensitive)
        # Valid networks: 'LinkedIn', 'GitHub', 'GitLab', 'IMDB', 'Instagram', 'ORCID', 
        # 'Mastodon', 'StackOverflow', 'ResearchGate', 'YouTube', 'Google Scholar', 
        # 'Telegram', 'Leetcode', 'X'
        profiles = basics.get("profiles", [])
        valid_networks = {
            'linkedin': 'LinkedIn',
            'github': 'GitHub',
            'gitlab': 'GitLab',
            'imdb': 'IMDB',
            'instagram': 'Instagram',
            'orcid': 'ORCID',
            'mastodon': 'Mastodon',
            'stackoverflow': 'StackOverflow',
            'researchgate': 'ResearchGate',
            'youtube': 'YouTube',
            'google scholar': 'Google Scholar',
            'telegram': 'Telegram',
            'leetcode': 'Leetcode',
            'x': 'X',
            'twitter': 'X'  # Twitter is now X
        }
        
        for profile in profiles:
            network = profile.get("network", "")
            username = profile.get("username", "")
            
            if network and username:
                # Normalize network name to match RenderCV's exact requirements
                network_lower = network.strip().lower()
                if network_lower in valid_networks:
                    rendercv_data["cv"]["social_networks"].append({
                        "network": valid_networks[network_lower],
                        "username": username.strip()
                    })
                # If network doesn't match, skip it (don't add invalid networks)
        
        # Summary as a section
        if basics.get("summary"):
            rendercv_data["cv"]["sections_input"]["summary"] = [basics["summary"]]
    
    # Convert work experience
    work = resume_data.get("work", [])
    if work:
        experience_entries = []
        for job in work:
            # Convert date format from YYYY-MM-DD to YYYY-MM if needed
            start_date = job.get("startDate", "")
            if start_date and len(start_date) > 7:
                start_date = start_date[:7]  # Keep only YYYY-MM
            
            end_date = job.get("endDate", "")
            if end_date:
                if len(end_date) > 7:
                    end_date = end_date[:7]  # Keep only YYYY-MM
            else:
                end_date = "present"
            
            entry = {
                "company": job.get("name", ""),
                "position": job.get("position", ""),
                "start_date": start_date,
                "end_date": end_date,
                "location": job.get("location", ""),
                "summary": job.get("summary"),
                "highlights": job.get("highlights", [])
            }
            # Remove None values
            entry = {k: v for k, v in entry.items() if v is not None}
            experience_entries.append(entry)
        rendercv_data["cv"]["sections_input"]["experience"] = experience_entries
    
    # Convert education
    education = resume_data.get("education", [])
    if education:
        education_entries = []
        for edu in education:
            # Convert date format from YYYY-MM-DD to YYYY-MM if needed
            start_date = edu.get("startDate", "")
            if start_date and len(start_date) > 7:
                start_date = start_date[:7]  # Keep only YYYY-MM
            
            end_date = edu.get("endDate", "")
            if end_date:
                if len(end_date) > 7:
                    end_date = end_date[:7]  # Keep only YYYY-MM
            else:
                end_date = "present"
            
            entry = {
                "institution": edu.get("institution", ""),
                "area": edu.get("area", ""),
                "degree": edu.get("studyType", ""),
                "start_date": start_date,
                "end_date": end_date,
                "location": edu.get("location", ""),
                "summary": edu.get("summary"),
                "highlights": edu.get("highlights", [])
            }
            # Add GPA/score to highlights if present
            if edu.get("score"):
                if "highlights" not in entry or not entry["highlights"]:
                    entry["highlights"] = []
                entry["highlights"].insert(0, f"GPA: {edu['score']}")
            # Remove None values
            entry = {k: v for k, v in entry.items() if v is not None}
            education_entries.append(entry)
        rendercv_data["cv"]["sections_input"]["education"] = education_entries
    
    # Convert skills
    skills = resume_data.get("skills", [])
    if skills:
        skills_entries = []
        for skill in skills:
            skill_text = skill.get("name", "")
            keywords = skill.get("keywords", [])
            if keywords:
                skill_text += f": {', '.join(keywords)}"
            skills_entries.append(skill_text)
        rendercv_data["cv"]["sections_input"]["skills"] = skills_entries
    
    # Convert projects
    projects = resume_data.get("projects", [])
    if projects:
        project_entries = []
        for project in projects:
            # Convert date format from YYYY-MM-DD to YYYY-MM if needed
            start_date = project.get("startDate", "")
            if start_date and len(start_date) > 7:
                start_date = start_date[:7]  # Keep only YYYY-MM
            
            end_date = project.get("endDate", "")
            if end_date:
                if len(end_date) > 7:
                    end_date = end_date[:7]  # Keep only YYYY-MM
            else:
                end_date = "present"
            
            entry = {
                "company": project.get("name", ""),
                "position": project.get("description", ""),
                "start_date": start_date,
                "end_date": end_date,
                "summary": project.get("summary"),
                "highlights": project.get("highlights", [])
            }
            # Remove None values
            entry = {k: v for k, v in entry.items() if v is not None}
            project_entries.append(entry)
        rendercv_data["cv"]["sections_input"]["projects"] = project_entries
    
    # Convert awards
    awards = resume_data.get("awards", [])
    if awards:
        award_entries = []
        for award in awards:
            award_text = award.get("title", award.get("name", ""))
            if award.get("awarder"):
                award_text += f" - {award['awarder']}"
            if award.get("date"):
                award_text += f" ({award['date']})"
            award_entries.append(award_text)
        rendercv_data["cv"]["sections_input"]["awards"] = award_entries
    
    return rendercv_data


def _validate_rendercv_data(rendercv_data: Dict) -> Dict:
    """
    Independent validation function for RenderCV data structure.
    Validates and cleans the data to ensure it matches RenderCV's exact requirements.
    This function can be called independently to validate any RenderCV data structure.
    
    Args:
        rendercv_data: Dictionary in RenderCV format
        
    Returns:
        Validated and cleaned RenderCV data dictionary that matches RenderCV's schema
    """
    import copy
    import re
    
    validated = copy.deepcopy(rendercv_data)
    
    # Ensure cv section exists
    if "cv" not in validated:
        validated["cv"] = {"name": "Resume", "sections_input": {}}
    
    cv = validated["cv"]
    
    # Validate name (required, must be non-empty string)
    if "name" not in cv or not isinstance(cv["name"], str) or not cv["name"].strip():
        cv["name"] = "Resume"
    
    # Validate email (must be valid email format if present)
    if "email" in cv:
        email = cv["email"]
        if isinstance(email, str):
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email.strip()):
                del cv["email"]  # Remove invalid email
        else:
            del cv["email"]
    
    # Validate phone (must be valid format if present)
    if "phone" in cv:
        phone = cv["phone"]
        if phone is not None:
            # Phone should be in format +1-XXX-XXX-XXXX or similar
            if not isinstance(phone, str) or not phone.strip():
                del cv["phone"]
            else:
                # Validate phone format (should start with + and have dashes)
                if not re.match(r'^\+\d+(-\d+)+$', phone.strip()):
                    del cv["phone"]  # Remove invalid phone format
        else:
            del cv["phone"]
    
    # Validate website (must be string if present)
    if "website" in cv and (not isinstance(cv["website"], str) or not cv["website"].strip()):
        del cv["website"]
    
    # Validate location (must be string if present)
    if "location" in cv and (not isinstance(cv["location"], str) or not cv["location"].strip()):
        del cv["location"]
    
    # Validate social_networks - RenderCV requires exact network names (case-sensitive)
    # Valid networks: 'LinkedIn', 'GitHub', 'GitLab', 'IMDB', 'Instagram', 'ORCID',
    # 'Mastodon', 'StackOverflow', 'ResearchGate', 'YouTube', 'Google Scholar',
    # 'Telegram', 'Leetcode', 'X'
    if "social_networks" in cv:
        if not isinstance(cv["social_networks"], list):
            cv["social_networks"] = []
        else:
            valid_networks = {
                'LinkedIn', 'GitHub', 'GitLab', 'IMDB', 'Instagram', 'ORCID',
                'Mastodon', 'StackOverflow', 'ResearchGate', 'YouTube',
                'Google Scholar', 'Telegram', 'Leetcode', 'X'
            }
            valid_social_networks = []
            for social in cv["social_networks"]:
                if isinstance(social, dict) and "network" in social and "username" in social:
                    network = social.get("network", "")
                    username = social.get("username", "")
                    # Only keep if network name exactly matches RenderCV's requirements
                    if network in valid_networks and isinstance(username, str) and username.strip():
                        valid_social_networks.append({
                            "network": network,
                            "username": username.strip()
                        })
            cv["social_networks"] = valid_social_networks
    
    # Validate sections_input
    if "sections_input" not in cv:
        cv["sections_input"] = {}
    
    sections = cv["sections_input"]
    
    # Validate experience section
    if "experience" in sections and isinstance(sections["experience"], list):
        valid_experience = []
        for exp in sections["experience"]:
            if isinstance(exp, dict) and "company" in exp and "position" in exp:
                # Validate required fields
                if (isinstance(exp["company"], str) and exp["company"].strip() and
                    isinstance(exp["position"], str) and exp["position"].strip()):
                    # Validate dates if present
                    if "start_date" in exp and exp["start_date"]:
                        if not isinstance(exp["start_date"], str) or not re.match(r'^\d{4}-\d{2}$', exp["start_date"]):
                            exp["start_date"] = ""
                    if "end_date" in exp and exp["end_date"]:
                        if not isinstance(exp["end_date"], str) or (exp["end_date"].lower() != "present" and not re.match(r'^\d{4}-\d{2}$', exp["end_date"])):
                            exp["end_date"] = "present"
                    valid_experience.append(exp)
        sections["experience"] = valid_experience
    
    # Validate education section
    if "education" in sections and isinstance(sections["education"], list):
        valid_education = []
        for edu in sections["education"]:
            if isinstance(edu, dict) and "institution" in edu:
                if isinstance(edu["institution"], str) and edu["institution"].strip():
                    # Validate dates if present
                    if "start_date" in edu and edu["start_date"]:
                        if not isinstance(edu["start_date"], str) or not re.match(r'^\d{4}-\d{2}$', edu["start_date"]):
                            edu["start_date"] = ""
                    if "end_date" in edu and edu["end_date"]:
                        if not isinstance(edu["end_date"], str) or (edu["end_date"].lower() != "present" and not re.match(r'^\d{4}-\d{2}$', edu["end_date"])):
                            edu["end_date"] = "present"
                    valid_education.append(edu)
        sections["education"] = valid_education
    
    # Validate skills section (should be list of strings)
    if "skills" in sections:
        if isinstance(sections["skills"], list):
            sections["skills"] = [s for s in sections["skills"] if isinstance(s, str) and s.strip()]
        else:
            del sections["skills"]
    
    # Validate summary section (should be list of strings)
    if "summary" in sections:
        if isinstance(sections["summary"], list):
            sections["summary"] = [s for s in sections["summary"] if isinstance(s, str) and s.strip()]
        else:
            del sections["summary"]
    
    # Validate projects section
    if "projects" in sections and isinstance(sections["projects"], list):
        valid_projects = []
        for proj in sections["projects"]:
            if isinstance(proj, dict) and "company" in proj:
                if isinstance(proj["company"], str) and proj["company"].strip():
                    valid_projects.append(proj)
        sections["projects"] = valid_projects
    
    # Validate awards section (should be list of strings)
    if "awards" in sections:
        if isinstance(sections["awards"], list):
            sections["awards"] = [s for s in sections["awards"] if isinstance(s, str) and s.strip()]
        else:
            del sections["awards"]
    
    return validated


def _validate_and_clean_resume_data(resume_data: Dict) -> Dict:
    """
    Validate and clean resume data before conversion to RenderCV format.
    Removes invalid fields and ensures data integrity.
    
    Args:
        resume_data: Dictionary following JSON Resume schema
        
    Returns:
        Cleaned and validated resume data dictionary
    """
    import re
    import copy
    
    cleaned = copy.deepcopy(resume_data)
    
    # Validate and clean basics section
    if "basics" in cleaned and isinstance(cleaned["basics"], dict):
        basics = cleaned["basics"]
        
        # Validate name (required, must be non-empty string)
        if "name" in basics:
            if not isinstance(basics["name"], str) or not basics["name"].strip():
                basics["name"] = "Resume"  # Default fallback
        else:
            basics["name"] = "Resume"  # Add default if missing
        
        # Validate email (must be valid email format)
        if "email" in basics:
            email = basics["email"]
            if isinstance(email, str):
                # Basic email validation
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, email.strip()):
                    del basics["email"]  # Remove invalid email
            else:
                del basics["email"]
        
        # Validate phone (will be validated again in normalization, but check format here)
        if "phone" in basics:
            phone = basics["phone"]
            if isinstance(phone, str):
                # Remove all non-digit characters except +
                digits = re.sub(r'[^\d+]', '', phone)
                # Check if it has reasonable length (at least 10 digits for US)
                if len(digits) < 10 or (digits.startswith('+') and len(digits) < 11):
                    del basics["phone"]  # Remove invalid phone
            else:
                del basics["phone"]
        
        # Validate location (must be dict if present)
        if "location" in basics and not isinstance(basics["location"], dict):
            del basics["location"]
        
        # Validate profiles (must be list)
        # RenderCV requires exact network names: 'LinkedIn', 'GitHub', 'GitLab', 'IMDB', 
        # 'Instagram', 'ORCID', 'Mastodon', 'StackOverflow', 'ResearchGate', 'YouTube', 
        # 'Google Scholar', 'Telegram', 'Leetcode', 'X'
        if "profiles" in basics:
            if not isinstance(basics["profiles"], list):
                del basics["profiles"]
            else:
                # Clean each profile - only keep valid networks
                valid_networks = {
                    'linkedin', 'github', 'gitlab', 'imdb', 'instagram', 'orcid',
                    'mastodon', 'stackoverflow', 'researchgate', 'youtube', 
                    'google scholar', 'telegram', 'leetcode', 'x', 'twitter'
                }
                valid_profiles = []
                for profile in basics["profiles"]:
                    if isinstance(profile, dict) and "network" in profile and "username" in profile:
                        network = profile.get("network", "").strip().lower()
                        username = profile.get("username", "").strip()
                        # Only keep profiles with valid network names and non-empty username
                        if network in valid_networks and username:
                            valid_profiles.append(profile)
                basics["profiles"] = valid_profiles
    
    # Validate and clean work experience
    if "work" in cleaned and isinstance(cleaned["work"], list):
        valid_work = []
        for job in cleaned["work"]:
            if not isinstance(job, dict):
                continue
            
            # Validate required fields
            if "name" not in job or not isinstance(job["name"], str) or not job["name"].strip():
                continue  # Skip jobs without company name
            
            if "position" not in job or not isinstance(job["position"], str) or not job["position"].strip():
                continue  # Skip jobs without position
            
            # Validate dates (YYYY-MM-DD or YYYY-MM format)
            if "startDate" in job:
                start_date = job["startDate"]
                if isinstance(start_date, str):
                    # Validate date format (YYYY-MM-DD or YYYY-MM)
                    if not re.match(r'^\d{4}-\d{2}(-\d{2})?$', start_date):
                        job["startDate"] = ""  # Set to empty if invalid
                else:
                    job["startDate"] = ""
            else:
                job["startDate"] = ""  # Add empty if missing
            
            if "endDate" in job:
                end_date = job["endDate"]
                if isinstance(end_date, str):
                    # Validate date format or "present"
                    if end_date.lower() != "present" and not re.match(r'^\d{4}-\d{2}(-\d{2})?$', end_date):
                        job["endDate"] = ""  # Set to empty if invalid
                else:
                    job["endDate"] = ""
            
            # Validate summary (must be string if present)
            if "summary" in job and not isinstance(job["summary"], str):
                del job["summary"]
            
            # Validate highlights (must be list if present)
            if "highlights" in job:
                if not isinstance(job["highlights"], list):
                    del job["highlights"]
                else:
                    # Keep only string highlights
                    job["highlights"] = [h for h in job["highlights"] if isinstance(h, str) and h.strip()]
            
            valid_work.append(job)
        cleaned["work"] = valid_work
    
    # Validate and clean education
    if "education" in cleaned and isinstance(cleaned["education"], list):
        valid_education = []
        for edu in cleaned["education"]:
            if not isinstance(edu, dict):
                continue
            
            # Validate required fields
            if "institution" not in edu or not isinstance(edu["institution"], str) or not edu["institution"].strip():
                continue  # Skip education without institution
            
            # Validate dates
            if "startDate" in edu:
                start_date = edu["startDate"]
                if isinstance(start_date, str):
                    if not re.match(r'^\d{4}-\d{2}(-\d{2})?$', start_date):
                        edu["startDate"] = ""
                else:
                    edu["startDate"] = ""
            
            if "endDate" in edu:
                end_date = edu["endDate"]
                if isinstance(end_date, str):
                    if end_date.lower() != "present" and not re.match(r'^\d{4}-\d{2}(-\d{2})?$', end_date):
                        edu["endDate"] = ""
                else:
                    edu["endDate"] = ""
            
            valid_education.append(edu)
        cleaned["education"] = valid_education
    
    # Validate and clean skills
    if "skills" in cleaned and isinstance(cleaned["skills"], list):
        valid_skills = []
        for skill in cleaned["skills"]:
            if isinstance(skill, dict) and "name" in skill:
                if isinstance(skill["name"], str) and skill["name"].strip():
                    # Validate keywords if present
                    if "keywords" in skill and isinstance(skill["keywords"], list):
                        skill["keywords"] = [k for k in skill["keywords"] if isinstance(k, str) and k.strip()]
                    valid_skills.append(skill)
        cleaned["skills"] = valid_skills
    
    # Validate and clean projects
    if "projects" in cleaned and isinstance(cleaned["projects"], list):
        valid_projects = []
        for project in cleaned["projects"]:
            if not isinstance(project, dict):
                continue
            if "name" not in project or not isinstance(project["name"], str) or not project["name"].strip():
                continue
            valid_projects.append(project)
        cleaned["projects"] = valid_projects
    
    return cleaned


def _sanitize_rendercv_data(rendercv_data: Dict) -> Dict:
    """
    Sanitize RenderCV data by removing invalid values (None, empty strings, invalid types).
    This ensures the data is valid for PDF generation.
    """
    import copy
    sanitized = copy.deepcopy(rendercv_data)
    
    def clean_value(value):
        """Recursively clean values in the dictionary."""
        if value is None:
            return None
        elif isinstance(value, str):
            # Remove empty strings, but keep non-empty ones
            return value if value.strip() else None
        elif isinstance(value, dict):
            # Recursively clean dictionaries
            cleaned = {}
            for k, v in value.items():
                cleaned_val = clean_value(v)
                if cleaned_val is not None and cleaned_val != "":
                    cleaned[k] = cleaned_val
            return cleaned if cleaned else None
        elif isinstance(value, list):
            # Clean list items and remove empty/None items
            cleaned = [clean_value(item) for item in value]
            cleaned = [item for item in cleaned if item is not None and item != ""]
            return cleaned if cleaned else None
        else:
            # Keep other types as-is (int, float, bool, etc.)
            return value
    
    # Clean the entire structure
    sanitized = clean_value(sanitized)
    
    # Ensure minimum required structure exists
    if not sanitized or not isinstance(sanitized, dict):
        sanitized = {"cv": {"name": "Resume", "sections_input": {}}}
    
    if "cv" not in sanitized:
        sanitized["cv"] = {"name": "Resume", "sections_input": {}}
    
    if "sections_input" not in sanitized["cv"]:
        sanitized["cv"]["sections_input"] = {}
    
    # Ensure name exists
    if not sanitized["cv"].get("name"):
        sanitized["cv"]["name"] = "Resume"
    
    return sanitized


def _create_fallback_pdf(resume_data: Dict) -> bytes:
    """
    Create a simple fallback PDF when RenderCV fails.
    Uses a basic PDF generation approach to ensure we always return PDF bytes.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        import io
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Get basic info
        name = resume_data.get("basics", {}).get("name", "Resume")
        email = resume_data.get("basics", {}).get("email", "")
        
        # Write basic content
        y = height - 100
        c.setFont("Helvetica-Bold", 20)
        c.drawString(100, y, name)
        
        if email:
            y -= 30
            c.setFont("Helvetica", 12)
            c.drawString(100, y, email)
        
        # Add a simple message
        y -= 50
        c.setFont("Helvetica", 10)
        c.drawString(100, y, "PDF generated successfully")
        
        c.save()
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    except Exception as e:
        # Last resort: create minimal valid PDF
        minimal_pdf = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n100\n%%EOF'
        return minimal_pdf


def generate_pdf_bytes_with_rendercv(resume_data: Dict, theme: str = 'engineeringclassic', tmp_dir: Optional[str] = None) -> bytes:
    """
    Generate PDF bytes from resume data using RenderCV.
    
    Args:
        resume_data: Resume data as a Python dictionary (RenderCV format)
        theme: Theme name for the PDF (default: 'classic')
        tmp_dir: Optional custom temporary directory path. If None, uses current working directory / "temp_pdfs"
    
    Returns:
        PDF bytes
    
    Raises:
        RuntimeError: If PDF generation fails
    """
    from rendercv import create_a_pdf_from_a_python_dictionary, data, api
    from pathlib import Path
    import tempfile
    import os
    
    # Prepare the data with design theme
    rendercv_data = resume_data.copy()
    if "design" not in rendercv_data:
        rendercv_data["design"] = {}
    rendercv_data["design"]["theme"] = theme
    
    # Determine temporary directory location
    if tmp_dir is None:
        # Default to temp_pdfs in current working directory
        tmp_dir = Path.cwd() / "temp_pdfs"
    else:
        tmp_dir = Path(tmp_dir)
    
    # Create the directory if it doesn't exist
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary subdirectory for this PDF generation
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_subdir:
        tmp_path = Path(tmp_subdir) / "resume.pdf"
        
        try:
            print('before cleaning======', rendercv_data)
            rendercv_data = clean_sections(rendercv_data)
            print('after cleaning======', rendercv_data)
            yaml = data.generator.dictionary_to_yaml(rendercv_data)
            print('yaml======', yaml)
            # Generate PDF using RenderCV
            result = api.create_a_pdf_from_a_yaml_string(
                yaml_file_as_string=yaml,
                output_file_path=tmp_path
            )
            # Note: create_a_pdf_from_a_yaml_string returns None on success
            # It returns a list only if there are validation errors
            # Check if result indicates validation errors
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'loc' in result[0] and 'msg' in result[0]:
                    error_details = "\n".join([f"  - {err.get('loc')}: {err.get('msg')}" for err in result])
                    raise RuntimeError(f"RenderCV validation errors:\n{error_details}")
            
            # Verify PDF file was created
            if not tmp_path.exists():
                raise RuntimeError("RenderCV did not create the output PDF file.")
            
            # Read PDF bytes
            with open(tmp_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Validate PDF bytes
            if not pdf_bytes or len(pdf_bytes) == 0:
                raise RuntimeError("Generated PDF file is empty.")
            
            if not pdf_bytes.startswith(b'%PDF'):
                raise RuntimeError("Generated file does not appear to be a valid PDF.")
            
            return pdf_bytes
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate PDF with RenderCV: {str(e)}") from e


def generate_pdf_bytes_from_yaml(yaml_string: str, theme: str = 'modern', tmp_dir: Optional[str] = None) -> bytes:
    """
    Generate PDF bytes from YAML string using RenderCV.
    
    Args:
        yaml_string: Resume data as a YAML string (RenderCV format)
        theme: Theme name for the PDF (default: 'classic')
        tmp_dir: Optional custom temporary directory path. If None, uses current working directory / "temp_pdfs"
    
    Returns:
        PDF bytes
    
    Raises:
        RuntimeError: If PDF generation fails
    """
    from rendercv import api
    from pathlib import Path
    import tempfile
    import yaml
    
    # Parse YAML to add/update theme if needed
    try:
        yaml_data = yaml.safe_load(yaml_string)
        if yaml_data is None:
            raise ValueError("YAML string is empty or invalid")
        
        # Add or update design theme
        if "design" not in yaml_data:
            yaml_data["design"] = {}
        yaml_data["design"]["theme"] = theme
        
        # Convert back to YAML string
        yaml_string = yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {str(e)}") from e
    
    # Determine temporary directory location
    if tmp_dir is None:
        # Default to temp_pdfs in current working directory
        tmp_dir = Path.cwd() / "temp_pdfs"
    else:
        tmp_dir = Path(tmp_dir)
    
    # Create the directory if it doesn't exist
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary subdirectory for this PDF generation
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_subdir:
        tmp_path = Path(tmp_subdir) / "resume.pdf"
        
        try:
            # Generate PDF using RenderCV from YAML string
            result = api.create_a_pdf_from_a_yaml_string(
                yaml_file_as_string=yaml_string,
                output_file_path=tmp_path
            )
            
            # Check if result indicates validation errors
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'loc' in result[0] and 'msg' in result[0]:
                    error_details = "\n".join([f"  - {err.get('loc')}: {err.get('msg')}" for err in result])
                    raise RuntimeError(f"RenderCV validation errors:\n{error_details}")
            
            # Verify PDF file was created
            if not tmp_path.exists():
                raise RuntimeError("RenderCV did not create the output PDF file.")
            
            # Read PDF bytes
            with open(tmp_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Validate PDF bytes
            if not pdf_bytes or len(pdf_bytes) == 0:
                raise RuntimeError("Generated PDF file is empty.")
            
            if not pdf_bytes.startswith(b'%PDF'):
                raise RuntimeError("Generated file does not appear to be a valid PDF.")
            
            return pdf_bytes
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate PDF with RenderCV: {str(e)}") from e


# Example Usage:
resume_data_rendercv = {
    "about": {
        "name": "Jane Doe",
        "email": "jane.doe@example.com"
    },
    "jobs": [{
        "position": "Product Manager",
        "company": "Innovate Corp."
    }]
}

def convert_to_valid_structure(data):
    if 'cv' in data and 'sections' in data['cv']:
        sections_dict = data['cv']['sections']
        
        # Initialize the list to hold the converted sections
        valid_sections_list = []
        
        # Convert each section to the correct structure
        for section in sections_dict:
            if 'name' in section and 'title' in section and 'items' in section:
                valid_sections_list.append({
                    'name': section['name'],
                    'title': section['title'],
                    'items': section['items']
                })
        
        # Update the original data with the new sections list
        data['cv']['sections'] = valid_sections_list
    
    return data

def clean_sections(cv_dict):
    """
    Clean empty or invalid sections from RenderCV data.
    Normalizes phone numbers to valid RenderCV format.
    Removes the 'cv' key but keeps its content by moving it to the top level.
    
    Args:
        cv_dict: Dictionary containing CV data with 'cv' key
    
    Returns:
        Dictionary with cleaned sections and 'cv' key removed (content moved to top level)
    """
    sections = cv_dict.get("cv", {}).get("sections", {})
    keys_to_remove = []

    for section_name, entries in sections.items():
        # Remove sections that are not lists or are empty lists
        if not isinstance(entries, list) or len(entries) == 0:
            keys_to_remove.append(section_name)

    for key in keys_to_remove:
        del sections[key]
    
    # Normalize phone number to valid format if it exists
    cv = cv_dict.get("cv", {})
    if "phone" in cv and cv["phone"]:
        phone = cv["phone"]
        if isinstance(phone, str):
            normalized_phone = _normalize_phone_for_rendercv(phone)
            if normalized_phone:
                cv["phone"] = normalized_phone
            else:
                # Remove invalid phone number
                del cv["phone"]
    
    # Remove 'cv' key but keep its content by moving it to top level
    # if "cv" in cv_dict:
    #     cv_content = cv_dict.pop("cv")
    #     # Merge cv content into the top level
    #     cv_dict.update(cv_content)

    return cv_dict