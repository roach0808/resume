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
You are a professional resume editor. Your task is to rewrite and expand the Experience section of a resume by doing the following:

Ensure every rewritten work experience bullet based on the current experiences, reflects 2–3 relevant tech stacks from the Extracted Tech Stacks.

If tech stacks, or others are missing from the original resume but appear in the extracted tech stack, generate new, enriched experience content by referencing the project list, tech context, and job description — making their use clear and contextually accurate.

Maintain or increase the length of the original experience sentences — do not shorten or simplify them.

Where appropriate, combine shorter bullets into longer, more technically rich and complete sentences.

🔻 Inputs:
1. Original Resume
Contains: name, job title, contact info, summary, experience bullets, and education
# original_resume
{resume_txt}

2. Extracted Tech Stacks
Flat list of all technologies, tools, libraries, platforms, and domains used by the candidate or required by the job
# extracted_tech_stacks
{extracted_tech_stacks}

3. Tech Context
Domain and focus of the candidate’s role (e.g., computer vision, MLOps, cloud infra)
# tech_context
{tech_context}

4. Project List
Detailed projects showing responsibilities, tools, and results
# projects
{projects}

5. Target Job Description
The job role this resume is being optimized for
# target_job_description
{target_job_description}

🎯 Instructions:
✅ 1. Header & Summary
Retain the candidate’s name, and contact info from the original resume.

update current job title to relevant job title in each companies with the target job title.

If a summary is present, update it to reflect the tech context and target role, keeping original tone and length.

✅ 2. Skills Section
Create a flat Skills section using all items from Extracted Tech Stacks.

Do not categorize (e.g., by language/tool/etc.) — just list them as-is, separated by commas.

✅ 3. Experience Section (Main Focus)
For each company in the experience section:
Use the original resume bullets, project list, tech context, and target job description to rewrite each bullet.

Each bullet must be equal to or longer than the original in sentence length and detail.

Integrate 2–3 relevant technologies from the Extracted Tech Stacks into each bullet.

If extracted tools like ComfyUI, Pillow, YOLOv8, or LangChain are not mentioned in the original experience, but appear in the project list or tech context:

Explicitly include them in the rewritten bullets

Generate realistic, accurate responsibilities or accomplishments that reflect their use, based on project list/tech context thorougly


Emphasize technical contributions, tools used, project impact, and measurable outcomes.

🔍 Example:
Extracted Tech Stack includes: ComfyUI, YOLOv8, Pillow, OpenCV
Original bullet:

Built automation scripts for image processing tasks.
Project list:

Built a pipeline using ComfyUI and YOLOv8 for object detection in satellite images.

Rewritten bullet (longer, enriched, integrated):

Designed and implemented automated image processing workflows using ComfyUI and Pillow, enabling object detection and classification from large satellite datasets using YOLOv8, significantly accelerating pipeline throughput.

✅ 4. Education Section
Retain the education section exactly as written in the original resume.

📤 Output Format:
Return a professionally formatted resume with:

Updated Summary (if present)

Flat Skills section (using all Extracted Tech Stacks)

Rewritten Experience section:

Each bullet includes 2–3 relevant tech stacks

All missing but relevant tech stacks (e.g., ComfyUI, Pillow) are inserted based on project/role context

Sentence length is equal to or longer than the original

Short bullets combined for clarity and impact

Unchanged Education section

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

