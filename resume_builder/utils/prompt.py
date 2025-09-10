parse_resume_prompt = """
You are a resume parser. Given the following raw resume text, extract:
- name
- experiences: a list of objects with company, duration, role (if mentioned), and bullet points (achievements/tasks)

Parse the following resume text and return the result strictly as plain JSON text with the following structure:
Only output the plain JSON text. Do not include any explanations, headers, or **markdown** formatting like ```json.
The contents not mentioned in the structure should be included in the "Extra" field.
The plain JSON text should be valid and **parsable**. Do not include any comments or explanations.

Don't summarize experiences in each company and don't split one bullit point into multiple bullit.
Maintain the resume structure and content.
Include detailed project information for each company with specific project names, descriptions, technologies used, and outcomes.
MANDATORY: Extract ALL professional experiences including InsoftAI, CoreWeave, Kensho, and Dana Scott Design.
For each company, include:
- A brief description of the company and your role
- Experience bullets using · symbol (5-7 detailed bullets)
- MANDATORY: Key project experience sections using - symbol with detailed project explanations and bullets using · symbol
- Each company MUST have multiple project sections (minimum 2-3 projects per company)
- Project sections must follow the EXACT format from original resume
- Update job titles to align with job requirements while maintaining similar seniority level

Note: The plain JSON text should not contain any extra spaces or new lines.

{{
  "name": "Full Name",
  "title": "Optional Title",
  "Contact": {{
    "email": "Email",
    "phone": "Phone Number",
    "location": "Location",
    "linkedin": "LinkedIn URL",
    "website": "Website URL",
  }}
  "summary": "Optional summary",
  "skills": {{
    "sub_skill_name": ['skill_1', 'skill_2', ...],
    ...,
  }},
  "experiences": [
    {{
      "company": "Company Name",
      "start_date": "Start Date",
      "end_date": "End Date",
      "role": "Optional Role",
      "location": "Company Location",
      "description": "Brief company description (1-2 lines)",
      "bullets": ["task 1", "task 2", ...],
      "project_sections": [
        {{
          "name": "Project Section Name",
          "description": "Brief project description",
          "bullets": ["project bullet 1", "project bullet 2", ...]
        }}
      ]
    }},
    ...
  ], 
  "education": [
    {{
      "institute_name": "Institute Name",
      "degree": "Degree",
      "start_date": "Start Date",
      "end_date": "End Date",
      "location": "University Location",
    }},
    ...
  ], 
  "Certificates": [
    {{
      "certificate_name": "Certificate Name",
      "issued_by": "Issuer",
      "issued_date": "Issued Date",
    }},
    ...
  ], 
  "Extra": {{
    "subtopic_1": {{
      "field_1": "value_1",
      "field_2": "value_2",
      ...
    }},
    ...,
    "subtopic_n": ["value_1", "value_2", ... ],
    ...,
    }}
  }}, 
}}

Resume text:
{text}
"""

job_description_prompt = """
You are a job description parser. Given the following job description, extract:
- company name
- company_summary: 2-line summary of what the company is doing now
- responsibilities: list of major duties
- required_skills: list of key technical requirements
    - skill: name of the skill
    - priority: important/nice-to-have for each

Parse the following job description and return the result strictly as plain JSON text with the following structure:
Only output plain JSON text. Do not include any explanations, headers, or **markdown** formatting like ```json.

Text:
{jd_text}
"""

skill_matching_prompt = """
You are an expert resume evaluator.

Given the following list of work experience bullet points and a skill requirement, analyze whether the experience fully covers the skill, partially covers it, or not-at-all. Your answer must be one of the following: fully, partially, or not at all.

Work Experience:
{work_experience}

Skill Requirement:
{skill_requirement}

Does the work experience above cover the skill requirement? Answer with one word: `fully`, `partially`, or `not-at-all`.
"""

easy_generate_prompt = """
You're helping rewriting the resume for a new role.

Job description (summary):
{jd_txt}

Current resume (summary):
{resume_txt}

Give me a updated and optimized ATS-friendly resume. 
The first thing you need to do is identify the key skills required and make sure those skills are included and reflected in the skills section and work experience. Be sure that if they require to have certain skill for certain years, you need to include that skill in the appropriate positions so that it can cover required period.
 - Please be sure to make every work experience unique and natural, realistic, professional by using real world projects in corresponding company in a way that don't list duties but includes handling which task using what technologies or tools and achieve which result like this.
 - If you find that the projects in my resume are great fit for the job description, then you should include them. If you can't find any relevant project in my original resume, then you should make up a real, natural-sounding work experience using your imagination and reflect its project name and the result. Just surprise me, I believe in you.
 - I mean, not explicitly specifying the project but reflecting project name in the work experience and the result with accurate mathematical result 
 - Increase the number of experience for each position to nearly 7 to 8  and hey remember to use unique and real world project names for many experience as possible.
 - It’s important to keep the number of experiences realistic given the time frame of each position. I'll adjust the number of experiences to 7-8 for each role while maintaining realism based on the amount of time you’ve spent in each position. I'll balance key accomplishments and responsibilities while making sure each project and experience fits into a reasonable timeline for each role. But always be sure to make most of experiences include the unique and real-sounding project name and its consequences.

 - Avoid a generic career summary.
 - In your bullet points, don't list responsibilities. List what you did to move the company or team forward, including key outcomes. Then, keep the achievements coming. The same idea applies to explaining your skills: Don't just make a list; talk about how you use them.
 - Incorporate metrics and details into your bullet points, focusing on your highlights, your "best of", your biggest achievements.
 - Use outcome metrics and scope of work to show your value and give context to your work.
 - Be sure to come up with project names that are natural-sounding while avoiding "X" like BridgeX or CrossChainX.
 - 
And be sure to include real world projects with github repository so afterwards I can show and explain in the upcoming technical interview.
 - When giving me the project name, be sure to make it real and existing project with actual github repository so I can explain it in the next technical interview.
 - Do not eliminate any section from original resume. Just update experiences, skills and summary to the existing ones.

Output only updated plain resume text with new bullets integrated without any description, header or markdown formattings.
"""

bullet_point_generator_prompt = """
Write {cnt} bullet points for the following resume.

Requirement: {requirement}

Current resume contents:
{experiences}

Your task:
1. Identify which required skills are missing from the resume.
2. For each missing skill, generate {cnt} bullet points that plausibly fit into different past companies.
3. Mention specific frameworks or models that were hot during that time (e.g. LangChain, Whisper, DeepSeek, etc.)
4. Must include quantified outcomes.

Output newly written bullet points.
[
  {{'company': 'Company Name', 'bullet': 'Bullet point'}},
]
"""

cover_letter_generator_prompt = """
You are a cover letter generator.
Given the following job description and resume, generate a professional cover letter, so that you could stand out as an exceptional candidate.
No markdown formatting, no code blocks, no explanations, just plain text.

Job Description:
{jd_info}

Resume:
{resume_json}
"""



def parse_job_description_prompt(jd_text):
    return job_description_prompt.format(jd_text=jd_text)

def get_skill_matching_prompt(work_experience, skill_requirement):
    return skill_matching_prompt.format(work_experience=work_experience, skill_requirement=skill_requirement)

def generate_bullet_points_prompt(experiences, requirement, cnt):
    return bullet_point_generator_prompt.format(experiences=experiences, requirement=requirement, cnt=cnt)

def generate_cover_letter_prompt(jd_info, resume_json):
    return cover_letter_generator_prompt.format(jd_info=jd_info, resume_json=resume_json)

def generate_easy_prompt(jd_txt, resume_txt):
    return easy_generate_prompt.format(jd_txt=jd_txt, resume_txt=resume_txt)