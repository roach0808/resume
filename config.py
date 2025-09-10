import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

system_tech_prt = """
Here I am going to have an online job interview with the HR manager on Machine learning Engineer role and in this call I have to demonstrate my previous experience with Go.
Be sure to give me the answers to the questions I specify in a natural, conversational, human, realistic, appealing, tactful, a bit humorous yet professional.
Explain about the Machine learning System architecture, pipelines, advanced RAG techniques, multi-indexing mechanism, chunking, CI/CD, and MLOps... and the other things related to job description.
and you can refer from the tech context provided below.
### Job descrioption:
{job_description}

### tech Context:
{context}


"""

# When answering to the question, please consider the following:
# ), internation arrow to speak like US native speaker for improving pronounciation, and mark the stress poing on the words in the sentence., When generate the answer, be sure to mark reading pause point on all sentences generated (e.g: "What ↗️ is // the best ↗️ approach ↘️ // to take ↗️ rest↘️?"

system_behavioral_prt = """
You are a senior AI/ML Engineer with more than 10 years of experience.
Please provide good answers that is best fit with job description to behavioral questions from the input .

## Job description:
{job_description}
"""

easy_generate_prompt = """
You are an expert resume optimization specialist. Your task is to create a highly targeted resume that achieves 99% alignment with the job description while preserving the candidate's authentic experience and adding comprehensive detail.

🔻 Inputs:
1. Original Resume
# original_resume
{resume_txt}

2. Extracted Tech Stacks
# extracted_tech_stacks
{extracted_tech_stacks}

3. Tech Context
# tech_context
{tech_context}

4. Project List
# projects
{projects}

5. Target Job Description
# target_job_description
{target_job_description}

🎯 CRITICAL REQUIREMENTS:

✅ 1. PRESERVE ALL ORIGINAL CONTENT
- Keep name, company names, project names EXACTLY as in original
- Retain ALL certificates section from original resume
- Maintain education section unchanged
- Preserve contact information
- UPDATE job titles for each company to match job requirements while maintaining similar level and scope
- MANDATORY: Include ALL 4 professional experiences: InsoftAI, CoreWeave, Kensho, Dana Scott Design

✅ 2. SKILLS SECTION - MAXIMUM JOB ALIGNMENT
- Create comprehensive skills section using ALL extracted tech stacks
- Prioritize skills by job description importance (most critical first)
- Add missing but relevant skills from job requirements
- Format as comma-separated list for ATS optimization

✅ 3. EXPERIENCE SECTION - DETAILED & COMPREHENSIVE
For EACH company/role, create 8-10 highly detailed bullet points that:

- Expand original experience with job-relevant technologies and methodologies
- Include specific metrics, achievements, and quantifiable results (use numbers, percentages, timeframes)
- Integrate 3-4 relevant tech stacks per bullet point with specific versions/frameworks
- Add detailed project explanations with technical depth and business impact
- Include specific project names, methodologies, and implementation details
- Emphasize leadership, ownership, technical innovation, and measurable outcomes
- Use strong action verbs and ATS-friendly language
- Ensure each bullet is comprehensive (3-4 sentences minimum with technical depth)
- Include specific technologies, tools, frameworks, and their business applications
- Add context about team size, budget impact, timeline, and scalability considerations

🎯 CRITICAL: PROJECT PRIORITIZATION STRATEGY
- For the MOST RECENT COMPANY (last company): Prioritize and expand job-relevant projects from original resume
- ONLY use projects that exist in the original resume - DO NOT create new projects
- If job requires healthcare experience, prioritize healthcare projects in the most recent company
- If job requires fintech experience, prioritize fintech projects in the most recent company
- Make the most recent company experience the most comprehensive and job-aligned
- For older companies, include relevant projects but focus more on the most recent company

✅ 4. PROJECT INTEGRATION & DETAILED PROJECT SECTIONS
- ONLY use projects from the original resume - DO NOT create new projects
- Prioritize job-relevant projects in the most recent company
- Ensure project descriptions match job requirements and are expanded with job-relevant details
- Maintain consistency with original resume's project style
- Add comprehensive project sections for each company with:
  * Project names and descriptions (from original resume)
  * Technical architecture and implementation details (expanded for job relevance)
  * Technologies used (with specific versions, aligned with job requirements)
  * Team collaboration and leadership aspects
  * Measurable outcomes and business impact
  * Timeline and budget considerations
  * Scalability and performance metrics

🔍 PROJECT SELECTION LOGIC:
- Analyze job description to identify key requirements (healthcare, fintech, AI/ML, etc.)
- For most recent company: Select and expand projects that best match job requirements
- For older companies: Include relevant projects but with less detail
- Ensure the most recent company has the most comprehensive and job-aligned project experience

✅ 5. CONTENT STYLE & FORMAT
- Use professional, concise language
- Ensure seamless integration without keyword stuffing
- Maintain consistent formatting and structure
- Keep language active and achievement-focused

✅ 6. JOB MATCHING STRATEGY
- Achieve 99% alignment with job requirements
- Incorporate ALL required and preferred skills
- Match domain knowledge and industry experience
- Align with company culture and role expectations

📤 OUTPUT FORMAT:
Return a complete, professionally formatted resume with:

1. Header (Name, Title, Contact Info)
2. Summary (Updated for job alignment)
3. Skills (Comprehensive, job-focused)
4. Professional Experience (Following original format structure)
5. Certificates (Preserved from original)
6. Education (Unchanged)

🎯 EXPERIENCE SECTION FORMAT (CRITICAL):
Follow this EXACT structure for each company:

Company Name, Location
Job Title (UPDATED to match job requirements)    Start Date – End Date
Brief company description (1-2 lines about the company's focus and your role)

🔍 JOB TITLE UPDATE STRATEGY:
- Analyze the target job description to identify the desired job title and level
- Update job titles for each company to align with job requirements while maintaining similar seniority level
- Examples:
  * If target job is "Senior AI/ML Engineer" → Update all company titles to similar AI/ML roles
  * If target job is "Healthcare AI Specialist" → Update titles to emphasize healthcare AI experience
  * If target job is "Cloud Solutions Architect" → Update titles to emphasize cloud architecture experience
- Maintain the progression and seniority level across companies
- Ensure titles reflect the actual work done at each company but align with job requirements

- Experience bullets using · symbol (5-7 detailed bullets)
· Detailed experience bullet 1 with specific technologies, metrics, and outcomes
· Detailed experience bullet 2 with specific technologies, metrics, and outcomes
· Detailed experience bullet 3 with specific technologies, metrics, and outcomes
· Detailed experience bullet 4 with specific technologies, metrics, and outcomes
· Detailed experience bullet 5 with specific technologies, metrics, and outcomes

- Project Section 1: Project Name - Brief Description
· Detailed project bullet 1 with technical implementation details
· Detailed project bullet 2 with specific technologies and outcomes
· Detailed project bullet 3 with metrics and business impact
· Detailed project bullet 4 with team collaboration and leadership aspects
· Detailed project bullet 5 with scalability and performance considerations

- Project Section 2: Project Name - Brief Description
· Detailed project bullet 1 with technical implementation details
· Detailed project bullet 2 with specific technologies and outcomes
· Detailed project bullet 3 with metrics and business impact
· Detailed project bullet 4 with team collaboration and leadership aspects
· Detailed project bullet 5 with scalability and performance considerations

🔍 EXAMPLE OF DETAILED EXPERIENCE BULLET:
Instead of: "Engineered multi-agent healthcare assistant systems for Doktor365, reducing clinician workload by 70% through CRM integration and AI-driven automation."

Write: "Architected and deployed a comprehensive multi-agent healthcare assistant platform for Doktor365, integrating specialized AI agents (secretary, clinic admin, tourism guide, pre-op coordinator) using LangGraph orchestration and real-time CRM synchronization. Implemented WhatsApp-based conversational interfaces with memory-persistent agents, enabling seamless patient intake workflows and automated appointment management. Developed robust webhook pipelines and event-driven architecture using FastAPI and WebSocket connections, achieving 70% reduction in operational workload while maintaining HIPAA compliance. Led a cross-functional team of 5 engineers over 8 months, delivering a production-grade solution that processed 10,000+ patient interactions monthly with 95% accuracy in intent recognition and automated 85% of routine administrative tasks."

🎯 PROJECT PRIORITIZATION EXAMPLE:
If job description requires "healthcare AI experience":
- MOST RECENT COMPANY (InsoftAI): Prioritize "Healthcare CRM-Integrated Multimodal Multi-Agent Healthcare Assistant System" project
- Expand this project with detailed technical implementation, metrics, and outcomes
- Include other relevant projects but focus primarily on healthcare-related ones
- For older companies: Include healthcare projects but with less detail

🎯 JOB TITLE UPDATE EXAMPLE:
If target job is "Senior Healthcare AI Engineer":
- InsoftAI: "Senior Healthcare AI Engineer & Multi-Agent Specialist" (instead of "Senior ML/MLOps Full Stack & AI-Agent Specialist")
- CoreWeave: "Senior AI Engineer - Healthcare & Fintech Solutions" (instead of "Senior AI/MLOps Engineer & Agent Developer")
- Kensho: "Senior AI Developer - Healthcare & Financial Systems" (instead of "Sr. Backend-heavy AI Developer")
- Dana Scott Design: "AI Research Intern - Healthcare & NLP Focus" (instead of "Full Stack AI Research Intern")

🔍 MANDATORY COMPANY INCLUSION:
You MUST include ALL 4 professional experiences in this exact order:
1. InsoftAI, FL, United State (MOST RECENT - prioritize job-relevant projects)
2. CoreWeave, Livingston, NJ (Detailed experience with job-aligned projects)
3. Kensho, Massachusetts, United State (Comprehensive experience with relevant projects)
4. Dana Scott Design, Indianapolis, United States (Complete experience with job-relevant projects)

🎯 MANDATORY PROJECT SECTIONS FOR EACH COMPANY:
For EVERY company experience, you MUST include:
- Key project experience sections using the EXACT format from original resume
- Project headers using "-" symbol followed by project name and description
- Project bullets using "·" symbol with detailed technical implementation
- Each company MUST have multiple project sections (minimum 2-3 projects per company)
- Projects must be job-relevant and prioritized based on job description requirements
- Follow the original resume's project section structure and formatting exactly

📋 PROJECT SECTION FORMAT EXAMPLE:
```
- Lead AI/Agent Engineer – Fintech-focused AI platform development
· Partnered with Kilocode.ai to co-develop their open-source AI code copilot platform...
· Integrated dynamic memory and context-awareness using MCP (Model Context Protocol)...
· Solved key growth challenges by building AI-powered analytics...

- Healthcare CRM-Integrated Multimodal Multi-Agent Healthcare Assistant System
· Architected and deployed a multi-agent assistant platform for Doktor365...
· Enabled WhatsApp-based conversational intake and follow-ups...
· Orchestrated multi-agent logic using LangGraph with dynamic state transitions...
```

Ensure the resume is ready for ATS systems and human review, with maximum impact and job relevance.
"""

# Second-generation prompt for improved results
resume_regeneration_prompt = """
You are an expert resume refinement specialist. I'm providing you with a previously generated resume that needs enhancement for maximum job alignment and professional presentation.

🔻 Inputs:
1. Previously Generated Resume
# previous_resume
{previous_resume}

2. Original Resume (for reference)
# original_resume
{original_resume}

3. Target Job Description
# target_job_description
{target_job_description}

4. Project List (for additional context)
# projects
{projects}

🎯 ENHANCEMENT REQUIREMENTS:

✅ 1. EXPERIENCE DETAIL ENHANCEMENT
- Make experience at ALL companies much more detailed (8-10 comprehensive bullet points per role)
- Add specific project experience with detailed explanations and technical depth
- Include quantifiable metrics, achievements, and technical outcomes (numbers, percentages, timeframes)
- Ensure each bullet point is substantial and impactful (3-4 sentences minimum)
- Match the detailed style of the original resume's project sections
- Add comprehensive project sections for each company with technical architecture details
- Include specific technologies, frameworks, and methodologies with versions
- Emphasize leadership, team collaboration, and measurable business impact

🎯 CRITICAL: PROJECT PRIORITIZATION FOR ENHANCEMENT
- For the MOST RECENT COMPANY: Prioritize and expand job-relevant projects from original resume
- ONLY use projects that exist in the original resume - DO NOT create new projects
- Focus on making the most recent company experience the most comprehensive and job-aligned
- If job requires specific domain experience (healthcare, fintech, etc.), prioritize those projects in the most recent company
- Ensure the most recent company has the most detailed and relevant project experience

🎯 JOB TITLE ENHANCEMENT:
- Update job titles for each company to better align with the target job requirements
- Maintain similar seniority level and progression across companies
- Ensure titles reflect the actual work done but emphasize job-relevant aspects
- Make titles more specific to the target job domain (healthcare, fintech, AI/ML, etc.)

🔍 MANDATORY COMPANY INCLUSION:
You MUST include ALL 4 professional experiences in this exact order:
1. InsoftAI, FL, United State (MOST RECENT - prioritize job-relevant projects)
2. CoreWeave, Livingston, NJ (Detailed experience with job-aligned projects)
3. Kensho, Massachusetts, United State (Comprehensive experience with relevant projects)
4. Dana Scott Design, Indianapolis, United States (Complete experience with job-relevant projects)

🎯 MANDATORY PROJECT SECTIONS FOR EACH COMPANY:
For EVERY company experience, you MUST include:
- Key project experience sections using the EXACT format from original resume
- Project headers using "-" symbol followed by project name and description
- Project bullets using "·" symbol with detailed technical implementation
- Each company MUST have multiple project sections (minimum 2-3 projects per company)
- Projects must be job-relevant and prioritized based on job description requirements
- Follow the original resume's project section structure and formatting exactly

✅ 2. TECHNICAL SKILLS OPTIMIZATION
- Ensure ALL technical skills perfectly match job description requirements
- Add any missing critical technologies from the job description
- Prioritize skills by job relevance and importance
- Include specific versions, frameworks, and methodologies mentioned in JD

✅ 3. CONTENT REFINEMENT
- Enhance language for maximum professional impact
- Add more specific technical details and methodologies
- Include industry-specific terminology and best practices
- Ensure ATS optimization while maintaining readability

✅ 4. FORMAT CONSISTENCY
- Maintain consistent formatting throughout
- Ensure proper section organization and flow
- Keep professional tone and structure
- Preserve all original information (name, companies, dates, etc.)

✅ 5. JOB ALIGNMENT MAXIMIZATION
- Achieve 99%+ alignment with job requirements
- Incorporate ALL required and preferred qualifications
- Match company culture and role expectations
- Ensure domain expertise is clearly demonstrated

📤 OUTPUT:
Return the enhanced, professional resume with maximum job alignment, detailed experience descriptions, and comprehensive technical coverage. The resume should be ready for immediate submission and optimized for both ATS systems and human reviewers.
"""

skill_extracting_prt = """
Please analyze the following job description and extract all relevant technical stack terms. Do not categorize them; simply provide a bullet list of the relevant tech stack names. Additionally, supplement the list with any relevant technical stacks based on your knowledge.

## Job Description:
{job_description}

Please format the output as a bullet list, ensuring clarity and organization."
"""

projects_txt = """

You can use these my real projects to build experience in new resume.
### building LLM Twin
It is an AI character that learns to write like somebody by incorporating its style and personality into an LLM.

### Financial advisor platform
implemented a real-time feature pipeline that streams financial news into a vector DB deployed on AWS.

### Personalized recommendation system
I built a highly scalable and modular real-time personalized recommender on retail platform data.

We designed a strategy similar to what TikTok employs for short videos, which will be applied to H&M retail items.

We will present all the architectural patterns necessary for building an end-to-end TikTok-like personalized recommender for H&M fashion items, from feature engineering to model training to real-time serving.

### Automating Multi-Specialist Medical Diagnosis
Traditional medical diagnosis can be time-consuming and requires collaboration between different specialists. AI can help streamline this process by providing initial assessments based on medical reports, allowing doctors to focus on critical cases and improving efficiency. This project aims to:

### folderr.com
AI platform that customer can build their own multi-agent system
Output only updated plain resume text with new bullets integrated without any description, header or markdown formattings.

### HIPPA project - https://chartauditor.com/
The project has a goal of helping healthcare providers and other professionals detect and help identify healthcare compliance issues as they arise with the goal of delivering a detailed report on what is compliant and what is not, making it easier for healthcare providers to identify areas that need improvement and maintain regulatory compliance. 
It’s a system that simplifies the processing of ensuring compliance with state and insurance regulations for patient charts in the behavioral health field. 
The system works by first de-identifying the patient chart, which means removing any personal information that could identify the patient. then compares it to medical necessity guidelines and state guidelines to generate a detailed report.

### Credit default predictions
This project is to develop ML models to predict whether a customer will fail to repay their loan or credit card balance. This helps financial institutions assess credit risk, make informed lending decisions, and reduce the likelihood of financial losses by identifying potential defaulters in advance

"""

cover_letter_generator_prompt = """
    You are a cover letter generator.
    Given the following tech context, job description and resume, generate a professional cover letter, so that you could stand out as an exceptional candidate.
    No markdown formatting, no code blocks, no explanations, just plain text.
    And don't include any contact informations and recipient information and company name and [XXX] info in the cover letter.
    Just start like this.
    'I am senior AI/ML Engineer...'

    tech context:
    {context}

    Job Description:
    {jd_txt}

    Resume:
    {resume_json}
    """

