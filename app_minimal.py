# === Minimal Streamlit App - No Heavy ML Libraries ===
import os
import streamlit as st
import time
from config import get_openai_api_key

# === CRITICAL: Force CPU-only mode to prevent Windows segmentation faults ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Constants ===
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Functions ===

def resume_openai_call(messages):
    """OpenAI API call for resume generation"""
    try:
        from openai import OpenAI
        
        api_key = get_openai_api_key()
        client = OpenAI(api_key=api_key, timeout=120.0)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.6,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error - Resume openai call: {str(e)}"

def interview_openai_call(messages):
    """OpenAI API call for interview responses"""
    try:
        from openai import OpenAI
        
        api_key = get_openai_api_key()
        client = OpenAI(api_key=api_key, timeout=20.0)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
    except Exception as e:
        yield f"⚠️ Error - interview openai call: {str(e)}"

def display_message(message, sender="assistant"):
    """Display chat messages"""
    icon = "🤖" if sender == "assistant" else "👤"
    alignment = "assistant" if sender == "assistant" else "user"
    st.markdown(f"""
        <div class="chat-message {alignment}">
            <div class="icon">{icon}</div>
            <div class="text">{message}</div>
        </div>
    """, unsafe_allow_html=True)

# === Resume Builder UI ===

def render_resume_builder(api_key):
    st.header("📝 Resume Builder")
    
    st.info("""
    **How to use:**
    1. Select a resume template from the dropdown
    2. Enter a job description
    3. Click "Generate Resume" to create an optimized resume
    """)

    option = st.selectbox(
        "Select Resume Type:",
        options=["M", "H", "R", "A", "M_Mikus"],
        format_func=lambda x: f"Option {x}"
    )

    # Map options to resume paths
    resume_paths = {
        "M": "resume_builder/demo_resume/m_resume.txt",
        "H": "resume_builder/demo_resume/h_resume.txt",
        "R": "resume_builder/demo_resume/r_resume.txt",
        "A": "resume_builder/demo_resume/a_resume.txt",
        "M_Mikus": "resume_builder/demo_resume/m_m_resume.txt"
    }

    resume_path = resume_paths.get(option)

    # Load resume text
    if not os.path.exists(resume_path):
        st.error(f"❌ Resume file not found for option {option}.")
        return

    try:
        with open(resume_path, "r", encoding="utf-8", errors="ignore") as file:
            resume_txt = file.read()
            
        if not resume_txt.strip():
            st.error(f"❌ Resume file is empty: {resume_path}")
            return
            
    except Exception as e:
        st.error(f"Failed to read resume: {str(e)}")
        return

    # Job Description Input
    job_description = st.text_area("Enter Job Description:", height=150)

    # Generate Button
    if st.button("🚀 Generate Resume"):
        if not job_description.strip():
            st.warning("Please enter a job description.")
            return

        try:
            # Step 1: Extract skills
            with st.spinner("🔍 Extracting skills from job description..."):
                try:
                    from config import skill_extracting_prt
                    
                    message = [
                        {"role": "system", "content": "You are resume builder"},
                        {"role": "user", "content": skill_extracting_prt.format(job_description=job_description)}
                    ]

                    skills_txt = resume_openai_call(message)
                    if "Error" in skills_txt:
                        st.error(f"Failed to extract skills: {skills_txt}")
                        return
                    
                    st.success("✅ Skills extracted successfully!")
                except Exception as e:
                    st.error(f"Failed to extract skills: {str(e)}")
                    return

            # Step 2: Generate resume (without ChromaDB for now)
            with st.spinner("🤖 Generating optimized resume..."):
                try:
                    from config import easy_generate_prompt, projects_txt
                    
                    # Use empty context for now (no ChromaDB)
                    context = ""
                    
                    message = [
                        {"role": "system", "content": "You are resume builder"},
                        {"role": "user", "content": easy_generate_prompt.format(
                            tech_context=context, 
                            target_job_description=job_description, 
                            resume_txt=resume_txt, 
                            projects=projects_txt, 
                            extracted_tech_stacks=skills_txt
                        )}
                    ]

                    response = resume_openai_call(message)
                    if "Error" in response:
                        st.error(f"Failed to generate resume: {response}")
                        return
                    
                    st.success("✅ Resume generated successfully!")
                    
                    # Display the generated resume
                    st.subheader("Generated Resume:")
                    st.text_area("Resume Content:", value=response, height=400)
                    
                except Exception as e:
                    st.error(f"Failed to generate resume: {str(e)}")
                    return

        except Exception as e:
            st.error(f"❌ An error occurred during resume generation: {str(e)}")
            return

# === Interview UI ===

def render_interview_ui(api_key):
    st.header("🎯 Interview Assistant")
    
    st.info("""
    **How to use:**
    1. Enter a job description
    2. Choose between Tech or Behavioral interview modes
    3. Ask questions and get AI responses
    """)

    st.sidebar.title("Job Description")
    job_description = st.sidebar.text_area("Enter Job Description:", height=150)

    st.sidebar.title("Interview Type")
    interview_type = st.sidebar.radio("Select Interview Type:", options=["Tech Interview", "Behavioral Interview"])

    query_text = st.text_input("Your Question:", key="query_input")

    if query_text and api_key:
        start_time = time.time()

        if interview_type == "Tech Interview":
            try:
                from config import system_tech_prt
                
                # Use empty context for now (no ChromaDB)
                context = ""
                display_message(query_text, sender="user")
                st.markdown("<hr>", unsafe_allow_html=True)

                messages = [
                    {"role": "system", "content": system_tech_prt.format(job_description=job_description, context=context)},
                    {"role": "user", "content": f"Question: {query_text}"}
                ]

            except Exception as e:
                st.error(f"Failed to load tech interview prompt: {str(e)}")
                return

        elif interview_type == "Behavioral Interview":
            try:
                from config import system_behavioral_prt
                
                display_message(query_text, sender="user")
                st.markdown("<hr>", unsafe_allow_html=True)

                messages = [
                    {"role": "system", "content": system_behavioral_prt.format(job_description=job_description)},
                    {"role": "user", "content": f"Question: {query_text}"}
                ]

            except Exception as e:
                st.error(f"Failed to load behavioral interview prompt: {str(e)}")
                return

        response_placeholder = st.empty()
        streamed_response = ""
        try:
            for chunk in interview_openai_call(messages):
                streamed_response += chunk
                response_placeholder.markdown(
                    f"<div style='font-size:18px; line-height:1.6; max-width: 400px; padding: 20px;'>{streamed_response}▌</div>",
                    unsafe_allow_html=True
                )

            response_placeholder.markdown(
                f"<div style='font-size:18px; line-height:1.6; max-width: 400px; padding: 20px;'>{streamed_response}</div>",
                unsafe_allow_html=True
            )
            end_time = time.time()
            print(f"Response time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            st.error(f"🚨 Streaming Error: {str(e)}")
            return

        st.session_state.chat_history.extend([
            {"role": "user", "content": query_text},
            {"role": "assistant", "content": streamed_response}
        ])

# === App Entry Point ===

def main():
    try:
        st.set_page_config(page_title="AI Career Assistant", layout="wide")
    except Exception as e:
        st.error(f"❌ Failed to configure Streamlit page: {str(e)}")
        return
        
    st.sidebar.title("🧭 Navigation")

    # Get API key from Streamlit secrets or environment variable
    try:
        api_key = get_openai_api_key()
        # Show API key status
        st.sidebar.success("✅ OpenAI API Key loaded")
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        st.info("💡 For cloud deployment: Set OPENAI_API_KEY in Streamlit secrets. For local: Set OPENAI_API_KEY environment variable.")
        return
    
    # Add startup health check
    try:
        import sys
        import gc
        st.sidebar.info(f"🐍 Python {sys.version.split()[0]}")
        st.sidebar.info(f"💾 Memory: {gc.get_count()}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ System check failed: {str(e)}")
    
    app_mode = st.sidebar.radio("Select App Mode:", options=["Interview", "Resume Builder"])

    # Show relevant content in main area
    if app_mode == "Interview":
        render_interview_ui(api_key)
    elif app_mode == "Resume Builder":
        render_resume_builder(api_key)

if __name__ == "__main__":
    main()
