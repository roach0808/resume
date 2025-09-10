import os
import json
import re
from typing import Dict, List
from docx import Document
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import warnings

from resume_builder.utils.prompt import parse_resume_prompt

# Suppress specific warnings from pdfplumber
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")


load_dotenv()

def _fix_json_syntax(json_str: str) -> str:
    """Fix common JSON syntax issues."""
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix missing commas between object properties
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
    json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)
    
    # Fix unescaped quotes in strings
    json_str = re.sub(r'(?<!\\)"(?=.*":)', r'\\"', json_str)
    
    return json_str

def _extract_json_with_regex(response: str) -> Dict:
    """Extract JSON data using regex patterns as a last resort."""
    result = {}
    
    # Extract name
    name_match = re.search(r'"name"\s*:\s*"([^"]*)"', response)
    if name_match:
        result['name'] = name_match.group(1)
    
    # Extract title
    title_match = re.search(r'"title"\s*:\s*"([^"]*)"', response)
    if title_match:
        result['title'] = title_match.group(1)
    
    # Extract summary
    summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', response)
    if summary_match:
        result['summary'] = summary_match.group(1)
    
    # Extract experiences (simplified)
    experiences = []
    exp_pattern = r'"company"\s*:\s*"([^"]*)"[^}]*"role"\s*:\s*"([^"]*)"[^}]*"start_date"\s*:\s*"([^"]*)"[^}]*"end_date"\s*:\s*"([^"]*)"'
    exp_matches = re.findall(exp_pattern, response, re.DOTALL)
    
    for match in exp_matches:
        company, role, start_date, end_date = match
        experiences.append({
            'company': company,
            'role': role,
            'start_date': start_date,
            'end_date': end_date,
            'bullets': []
        })
    
    if experiences:
        result['experiences'] = experiences
    
    # Extract skills (simplified)
    skills_match = re.search(r'"skills"\s*:\s*\{([^}]*)\}', response, re.DOTALL)
    if skills_match:
        result['skills'] = {}
    
    # Extract education (simplified)
    education = []
    edu_pattern = r'"institute_name"\s*:\s*"([^"]*)"[^}]*"degree"\s*:\s*"([^"]*)"'
    edu_matches = re.findall(edu_pattern, response, re.DOTALL)
    
    for match in edu_matches:
        institute, degree = match
        education.append({
            'institute_name': institute,
            'degree': degree,
            'start_date': '',
            'end_date': ''
        })
    
    if education:
        result['education'] = education
    
    return result

def resume_openai_call(messages):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, timeout=20.0)
    try:        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except (ConnectionError, TimeoutError, ValueError) as e:
        return f"⚠️ Error - Resume openai call.: {str(e)}"

def read_docx_text(path: str) -> List[str]:
    doc = Document(path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    text = "\n".join(lines)
    return text

def parse_docx_resume(path: str) -> Dict:
    return parse_text_resume(read_docx_text(path))

def read_pdf_text(path: str) -> List[str]:
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines += page.extract_text().split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    text = "\n".join(lines)

    return text

def parse_pdf_resume(path: str) -> Dict:
    return parse_text_resume(read_pdf_text(path))

def parse_text_resume(text: str) -> Dict:
    # prompt = parse_resume_prompt(text)
    print("resume text", text)
    messages = [
        {"role": "system", "content": "You are resume parser assistant"},
        {"role": "user", "content": parse_resume_prompt.format(text=text)}
        ]

    response= resume_openai_call(messages)
    print("parsed resume response", response)
    # response = llm_inference(prompt, temperature=0.0, max_tokens=2000)
    if response is None:
        raise ValueError("No response from LLM.")
    
    # Parse the JSON response with multiple fallback strategies
    try:
        # Remove any markdown formatting if present
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        elif response.startswith("```"):
            response = response.replace("```", "").strip()
        
        # Strategy 1: Try to parse the entire response as JSON
        try:
            parsed_response = json.loads(response)
            return parsed_response
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find the first complete JSON object
        start_idx = response.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        
        # Count braces to find the end of the first JSON object
        brace_count = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        
        for i, char in enumerate(response[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
        
        # Extract only the first JSON object
        json_str = response[start_idx:end_idx]
        
        # Strategy 3: Try to fix common JSON issues
        try:
            parsed_response = json.loads(json_str)
            return parsed_response
        except json.JSONDecodeError as e:
            # Try to fix common JSON syntax issues
            fixed_json = _fix_json_syntax(json_str)
            try:
                parsed_response = json.loads(fixed_json)
                return parsed_response
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Use regex to extract key-value pairs and reconstruct JSON
        try:
            parsed_response = _extract_json_with_regex(response)
            return parsed_response
        except Exception:
            pass
        
        # If all strategies fail, raise an error
        print(f"All JSON parsing strategies failed")
        print(f"Raw response: {response}")
        raise ValueError("Failed to parse JSON response with all available strategies")
        
    except (ValueError, IndexError) as e:
        print(f"Unexpected error during JSON parsing: {e}")
        print(f"Raw response: {response}")
        raise ValueError(f"Failed to parse JSON response: {e}") from e

def parse_resume(path: str) -> Dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return parse_docx_resume(path)
    elif ext == ".pdf":
        return parse_pdf_resume(path)
    else:
        raise ValueError("Unsupported file type")

if __name__ == "__main__":
    # Example usage
    resume_path = "resume_builder/demo_resume/bobby.pdf"  # Change this to your resume path
    parsed_resume = parse_resume(resume_path)
    # dump json to file
    with open("parsed_resume.json", "w", encoding="utf-8") as f:
        json.dump(parsed_resume, f, indent=4)
    # print parsed resume
    print(parsed_resume)