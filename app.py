# === Standard Imports ===
import os
import shutil
import streamlit as st
import time
from dotenv import load_dotenv

# === CRITICAL: Force CPU-only mode to prevent Windows segmentation faults ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warnings

# Load environment variables
load_dotenv()

# === Lazy Imports (moved to functions to prevent startup crashes) ===
# Heavy imports like chromadb, sentence-transformers, langchain are now imported only when needed

# === Constants ===
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "paraphrase-MiniLM-L3-v2"  # Lighter model to prevent crashes
COLLECTION_NAME = "pdf_chunks"

# Global variables for lazy loading
client = None
collection = None
embedding_func = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Functions ===

def initialize_chromadb():
    """Initialize ChromaDB components only when needed with better error handling"""
    global client, collection, embedding_func
    
    if client is None:
        try:
            st.write("🔧 Starting ChromaDB initialization...")
            
            # Lazy import to prevent startup crashes
            st.write("📦 Importing ChromaDB...")
            import chromadb
            from chromadb.utils import embedding_functions
            import gc  # For garbage collection
            st.write("✅ ChromaDB imports successful")
            
            # Ensure the directory exists
            os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
            
            with st.spinner("Initializing ChromaDB..."):
                # Create a fresh client with error handling
                try:
                    st.write("🔧 Creating ChromaDB client...")
                    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
                    st.write("✅ ChromaDB client created")
                except Exception as client_error:
                    st.error(f"❌ Failed to create ChromaDB client: {str(client_error)}")
                    return None, None, None
                
                # Initialize embedding function with memory management
                try:
                    st.write("🤖 Initializing OpenAI embeddings...")
                    # Use OpenAI embeddings instead of sentence transformers
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        st.error("❌ OpenAI API key not found for embeddings")
                        return None, None, None
                    
                    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        model_name="text-embedding-3-small"  # Fast and cheap
                    )
                    st.write("✅ OpenAI embeddings initialized")
                except Exception as embed_error:
                    st.error(f"❌ Failed to initialize embedding function: {str(embed_error)}")
                    st.info("💡 This might be due to insufficient memory. Try restarting the app.")
                    return None, None, None
                
                # Create or get collection with error handling
                try:
                    collection = client.get_or_create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=embedding_func,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception as collection_error:
                    st.error(f"❌ Failed to create collection: {str(collection_error)}")
                    return None, None, None
                
                # Force garbage collection to free memory
                gc.collect()
                st.success("✅ ChromaDB initialized successfully!")
                
        except ImportError as import_error:
            st.error(f"❌ Missing required packages: {str(import_error)}")
            st.info("💡 Please install required packages: pip install chromadb sentence-transformers")
            return None, None, None
        except MemoryError as memory_error:
            st.error(f"❌ Insufficient memory: {str(memory_error)}")
            st.info("💡 Try closing other applications or restarting your system.")
            return None, None, None
        except Exception as e:
            st.error(f"❌ ChromaDB initialization failed: {str(e)}")
            st.info("💡 You can still use the app, but PDF processing features will be limited.")
            return None, None, None
    
    return client, collection, embedding_func

def process_pdfs(file_paths, chunk_size, chunk_overlap):
    """Process PDF files with improved error handling and memory management"""
    try:
        # Lazy import to prevent startup crashes
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import gc  # For garbage collection
    except ImportError as import_error:
        st.error(f"❌ Missing required packages: {str(import_error)}")
        st.info("💡 Please install required packages: pip install langchain langchain-community")
        return 0
    
    client, collection, embedding_func = initialize_chromadb()
    
    if collection is None:
        st.error("❌ ChromaDB not available. Cannot process PDFs.")
        return 0
    
    chunk_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        status_text.text(f"📄 Processing {filename}...")
        
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                st.error(f"❌ File not found: {filename}")
                continue
                
            if not os.access(file_path, os.R_OK):
                st.error(f"❌ File not readable: {filename}")
                continue
            
            # Load PDF with error handling
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                if not documents:
                    st.warning(f"⚠️ No content found in {filename}")
                    continue
                    
            except Exception as pdf_error:
                st.error(f"❌ Failed to load PDF {filename}: {str(pdf_error)}")
                continue
            
            status_text.text(f"✂️ Chunking {filename} ({len(documents)} pages)...")
            
            # Split into chunks with error handling
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = text_splitter.split_documents(documents)
                
                if not chunks:
                    st.warning(f"⚠️ No chunks created from {filename}")
                    continue
                    
            except Exception as chunk_error:
                st.error(f"❌ Failed to chunk {filename}: {str(chunk_error)}")
                continue
            
            status_text.text(f"💾 Adding {len(chunks)} chunks to database...")
            
            # Prepare data for ChromaDB with memory management
            try:
                documents_texts = [doc.page_content for doc in chunks]
                metadatas = [doc.metadata for doc in chunks]
                
                # Add to ChromaDB in smaller batches to avoid memory issues
                batch_size = 50  # Reduced batch size for better memory management
                for j in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[j:j+batch_size]
                    batch_texts = documents_texts[j:j+batch_size]
                    batch_metadatas = metadatas[j:j+batch_size]
                    
                    try:
                        collection.add(
                            documents=batch_texts,
                            metadatas=batch_metadatas,
                            ids=[f"{filename}_id_{j+k}" for k in range(len(batch_chunks))]
                        )
                    except Exception as db_error:
                        st.error(f"❌ Failed to add batch to database: {str(db_error)}")
                        break
                    
                    # Force garbage collection after each batch
                    gc.collect()
                
                chunk_count += len(chunks)
                status_text.text(f"✅ {filename} processed successfully! ({len(chunks)} chunks)")
                
            except Exception as db_error:
                st.error(f"❌ Database error processing {filename}: {str(db_error)}")
                continue
            
        except MemoryError as memory_error:
            st.error(f"❌ Insufficient memory processing {filename}: {str(memory_error)}")
            st.info("💡 Try processing smaller files or restart the app.")
            break
        except Exception as e:
            status_text.text(f"❌ Error processing {filename}: {str(e)}")
            st.error(f"Failed to process {filename}: {str(e)}")
            continue
        
        # Update progress
        progress_bar.progress((i + 1) / len(file_paths))
    
    # Clear status messages
    status_text.empty()
    progress_bar.empty()
    
    # Final garbage collection
    gc.collect()
    
    return chunk_count

def query_chunks(query_text, top_n=5):
    client, collection, embedding_func = initialize_chromadb()
    
    if collection is None:
        return ["ChromaDB not available. Please upload PDFs first."]
    
    results = collection.query(
        query_texts=[query_text],
        n_results=top_n,
    )
    if results["documents"]:
        top_chunks = [doc for docs in results["documents"] for doc in docs][:top_n]
        return top_chunks
    return ["No relevant information found."]


def resume_openai_call(messages):
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, timeout=120.0)  # Increase timeout to 2 minutes
    retries = 2  # Reduce retries to prevent long waits
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.6,
                max_tokens=4000  # Limit response size to prevent memory issues
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:  # Retry if attempts are left
                time.sleep(3)  # Wait for 3 seconds before retrying
                continue
            return f"⚠️ Error - Resume openai call: {str(e)}"

def interview_openai_call(messages):
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, timeout=20.0)
    try:
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
    
    # Show instructions
    st.info("""
    **How to use:**
    1. Select a resume template from the dropdown
    2. Enter a job description
    3. Click "Generate Resume" to create an optimized resume
    4. Use "Generate Enhanced Resume" for even better results
    """)

    option = st.selectbox(
        "Select Resume Type:",
        options=["J", "B", "D"],
        format_func=lambda x: {
            "J": "John Thomason (insoftai, coreweave, kensho, dana scott design)",
            "B": "Bobby Estes (Fingent, TravelPerk, Voyage Privé, Amazon)",
            "D": "Davante Bonham (Fingent, Sigma AI, Kensho, Amazon.com)"
        }.get(x, f"Option {x}")
    )

    # Map options to resume paths
    resume_paths = {
        "J": "resume_builder/demo_resume/m_resume.txt",
        "B": "resume_builder/demo_resume/h_resume.txt",  # Bobby Estes
        "D": "resume_builder/demo_resume/r_resume.txt"   # Davante Bonham
    }
    
    # Company lists for specific resume types
    company_lists = {
        "J": [  # John Thomason companies
            "InsoftAI",
            "CoreWeave", 
            "Kensho Technologies",
            "Dana Scott Design"
        ],
        "B": [  # Bobby Estes companies
            "Fingent",
            "TravelPerk", 
            "Voyage Privé",
            "Amazon"
        ],
        "D": [  # Davante Bonham companies
            "Fingent",
            "Sigma AI",
            "Kensho Technologies", 
            "Amazon.com"
        ]
    }

    # Get the selected resume path
    resume_path = resume_paths.get(option)

    # === Load resume text ===
    if not os.path.exists(resume_path):
        st.error(f"❌ Resume file not found for option {option}.")
        st.info(f"💡 Expected file: {resume_path}")
        return

    try:
        with open(resume_path, "r", encoding="utf-8", errors="ignore") as file:
            resume_txt = file.read()  # Read the content of the .txt file
            
        if not resume_txt.strip():
            st.error(f"❌ Resume file is empty: {resume_path}")
            return
            
    except PermissionError:
        st.error(f"❌ Permission denied reading resume file: {resume_path}")
        return
    except MemoryError:
        st.error(f"❌ Insufficient memory to read resume file: {resume_path}")
        return
    except Exception as e:
        st.error(f"Failed to read resume: {str(e)}")
        return

    # === Job Description Input ===
    job_description = st.text_area("Enter Job Description:", height=150)

    # === Generate Button ===
    if st.button("🚀 Generate Resume"):
        if not job_description.strip():
            st.warning("Please enter a job description.")
            return

        try:
            # Step 1: Extract skills
            with st.spinner("🔍 Extracting skills from job description..."):
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

            # Step 2: Get context from ChromaDB
            with st.spinner("🔍 Matching job description with resume..."):
                top_chunks = query_chunks(job_description, top_n=5)
                context = "\n\n".join(top_chunks)
                st.success("✅ Context retrieved successfully!")

            # Step 3: Generate resume
            with st.spinner("🤖 Generating optimized resume..."):
                from config import easy_generate_prompt, projects_txt
                
                # Get specific company list for the selected resume type
                specific_companies = company_lists.get(option, [])
                companies_text = "\n".join([f"- {company}" for company in specific_companies]) if specific_companies else ""
                
                # Create enhanced prompt with specific companies and styling
                styling_instructions = """
                
🎨 STYLING REQUIREMENTS:
- Format company names with blue color: <span style="color: #0066cc; font-weight: bold;">Company Name</span>
- Format job titles with blue color: <span style="color: #0066cc; font-weight: bold;">Job Title</span>
- Use HTML formatting for better visual presentation
- Example: <span style="color: #0066cc; font-weight: bold;">Kensho Technologies</span>, <span style="color: #0066cc; font-weight: bold;">Senior ML Engineer</span>
"""
                
                enhanced_prompt = easy_generate_prompt + f"\n\nIMPORTANT: Use these specific companies in the experience section: {companies_text}" + styling_instructions
                
                message = [
                    {"role": "system", "content": "You are resume builder"},
                    {"role": "user", "content": enhanced_prompt.format(tech_context=context, target_job_description=job_description, resume_txt=resume_txt, 
                    projects=projects_txt, 
                    extracted_tech_stacks=skills_txt)}
                ]

                response = resume_openai_call(message)
                if "Error" in response:
                    st.error(f"Failed to generate resume: {response}")
                    return
                
                st.success("✅ Resume generated successfully!")

            # Step 4: Parse resume text
            with st.spinner("📝 Parsing resume structure..."):
                from resume_builder.parser.resume_parser import parse_text_resume
                parsed_resume_json = parse_text_resume(response)
                st.success("✅ Resume structure parsed successfully!")

            # Step 5: Generate cover letter
            with st.spinner("📄 Generating cover letter..."):
                from config import cover_letter_generator_prompt
                
                message = [
                    {"role": "system", "content": "You are cover letter builder"},
                    {"role": "user", "content": cover_letter_generator_prompt.format(context=context, jd_txt=job_description, resume_json=resume_txt)}
                ]
                cover_letter = resume_openai_call(message)
                if "Error" in cover_letter:
                    st.error(f"Failed to generate cover letter: {cover_letter}")
                    return
                st.success("✅ Cover letter generated successfully!")

            # Step 6: Generate final output
            with st.spinner("💾 Creating final documents..."):
                from resume_builder.autocv_core import generate_final_output
                
                output_dir = 'resume_builder/demo_resume/created_resume'
                result = generate_final_output(job_description, parsed_resume_json, cover_letter, output_dir, 'pdf', 'both')
                st.success("✅ Final documents created successfully!")
            
            # Store the generated resume for potential regeneration
            st.session_state.generated_resume = response
            st.session_state.original_resume = resume_txt
            st.session_state.job_description = job_description
            
            # Display the generated resume with HTML styling
            st.markdown("---")
            st.subheader("📄 Generated Resume")
            st.markdown(response, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ An error occurred during resume generation: {str(e)}")
            st.info("💡 Please try again with a shorter job description or check your internet connection.")
            return
        
        # Add regeneration button
        st.markdown("---")
        st.subheader("🔄 Resume Enhancement")
        st.info("💡 If you want to improve the resume further, click the button below for enhanced generation with more detailed experience and better job alignment.")
        
        if st.button("🚀 Generate Enhanced Resume (2nd Generation)"):
            if 'generated_resume' in st.session_state:
                try:
                    with st.spinner("🔄 Generating enhanced resume with improved detail and job alignment..."):
                        from config import resume_regeneration_prompt, projects_txt
                        
                        # Get specific company list for the selected resume type
                        specific_companies = company_lists.get(option, [])
                        companies_text = "\n".join([f"- {company}" for company in specific_companies]) if specific_companies else ""
                        
                        # Create enhanced prompt with specific companies and styling
                        styling_instructions = """
                        
🎨 STYLING REQUIREMENTS:
- Format company names with blue color: <span style="color: #0066cc; font-weight: bold;">Company Name</span>
- Format job titles with blue color: <span style="color: #0066cc; font-weight: bold;">Job Title</span>
- Use HTML formatting for better visual presentation
- Example: <span style="color: #0066cc; font-weight: bold;">Kensho Technologies</span>, <span style="color: #0066cc; font-weight: bold;">Senior ML Engineer</span>
"""
                        
                        enhanced_regeneration_prompt = resume_regeneration_prompt + f"\n\nIMPORTANT: Use these specific companies in the experience section: {companies_text}" + styling_instructions
                        
                        message = [
                            {"role": "system", "content": "You are an expert resume enhancement specialist"},
                            {"role": "user", "content": enhanced_regeneration_prompt.format(
                                previous_resume=st.session_state.generated_resume,
                                original_resume=st.session_state.original_resume,
                                target_job_description=st.session_state.job_description,
                                projects=projects_txt
                            )}
                        ]
                        
                        enhanced_response = resume_openai_call(message)
                        if "Error" in enhanced_response:
                            st.error(f"Failed to generate enhanced resume: {enhanced_response}")
                            return
                        
                        st.success("✅ Enhanced resume content generated!")

                    with st.spinner("📝 Parsing enhanced resume structure..."):
                        from resume_builder.parser.resume_parser import parse_text_resume
                        enhanced_parsed_resume_json = parse_text_resume(enhanced_response)
                        st.success("✅ Enhanced resume structure parsed!")

                    with st.spinner("📄 Generating enhanced cover letter..."):
                        from config import cover_letter_generator_prompt
                        
                        message = [
                            {"role": "system", "content": "You are cover letter builder"},
                            {"role": "user", "content": cover_letter_generator_prompt.format(context=context, jd_txt=job_description, resume_json=enhanced_response)}
                        ]
                        enhanced_cover_letter = resume_openai_call(message)
                        if "Error" in enhanced_cover_letter:
                            st.error(f"Failed to generate enhanced cover letter: {enhanced_cover_letter}")
                            return
                        st.success("✅ Enhanced cover letter generated!")

                    with st.spinner("💾 Creating enhanced final documents..."):
                        from resume_builder.autocv_core import generate_final_output
                        enhanced_result = generate_final_output(job_description, enhanced_parsed_resume_json, enhanced_cover_letter, output_dir, 'pdf', 'both')
                        st.success("✅ Enhanced final documents created!")
                    
                    st.success("🎉 Enhanced resume generated successfully!")
                    st.info("📄 The enhanced resume includes more detailed experience descriptions, better job alignment, and comprehensive technical coverage.")
                    
                    # Display the enhanced resume with HTML styling
                    st.markdown("---")
                    st.subheader("📄 Enhanced Resume")
                    st.markdown(enhanced_response, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during enhanced resume generation: {str(e)}")
                    st.info("💡 Please try again or check your internet connection.")
            else:
                st.warning("Please generate a resume first before using the enhancement feature.")
# === Interview UI ===

def render_interview_ui(api_key):
    st.header("🎯 Interview Assistant")
    
    # Show instructions
    st.info("""
    **How to use:**
    1. Click "Check ChromaDB Status" in the sidebar to initialize the database
    2. Upload PDF documents to build your knowledge base
    3. Enter a job description and ask questions
    4. Choose between Tech or Behavioral interview modes
    """)

    chunk_size = st.sidebar.slider("Chunk Size:", 100, 2000, 500, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap:", 0, 500, 50, 10)

    st.sidebar.title("Job Description")
    job_description = st.sidebar.text_area("Enter Job Description:", height=150)

    st.sidebar.title("Interview Type")
    interview_type = st.sidebar.radio("Select Interview Type:", options=["Tech Interview", "Behavioral Interview"])

    st.sidebar.title("ChromaDB Stats")
    if st.sidebar.button("🔄 Check ChromaDB Status", key="interview_chromadb"):
        try:
            client, collection, embedding_func = initialize_chromadb()
            if collection is not None:
                total_chunks = collection.count()
                st.sidebar.write(f"📊 Total Chunks: {total_chunks}")
            else:
                st.sidebar.warning("⚠️ ChromaDB not available")
        except Exception as e:
            st.sidebar.error(f"Error fetching chunk count: {str(e)}")
    else:
        st.sidebar.info("💡 Click 'Check ChromaDB Status' to initialize")

    st.sidebar.title("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files and api_key:
        # Check if ChromaDB is initialized
        client, collection, embedding_func = initialize_chromadb()
        if collection is None:
            st.error("❌ Please initialize ChromaDB first by clicking 'Check ChromaDB Status'")
        else:
            temp_file_paths = []
            st.info(f"📁 Preparing {len(uploaded_files)} file(s) for processing...")
            
            for uploaded_file in uploaded_files:
                temp_path = os.path.join("temp", uploaded_file.name)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_file_paths.append(temp_path)
                st.write(f"📄 Saved: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")

            st.info("🚀 Starting PDF processing and chunking...")
            num_chunks = process_pdfs(temp_file_paths, chunk_size, chunk_overlap)

            # Clean up temp files
            for path in temp_file_paths:
                try:
                    os.remove(path)
                except:
                    pass

            if num_chunks > 0:
                st.success(f"🎉 Successfully processed {len(uploaded_files)} PDF(s) into {num_chunks} chunks!")
                st.info("💡 You can now ask questions about the uploaded documents!")
            else:
                st.warning("⚠️ No chunks were created. Please check your PDF files.")

    query_text = st.text_input("Your Question:", key="query_input")

    if query_text and api_key:
        start_time = time.time()

        if interview_type == "Tech Interview":
            from config import system_tech_prt
            
            top_chunks = query_chunks(query_text, top_n=5)
            context = "\n\n".join(top_chunks)
            display_message(query_text, sender="user")
            st.markdown("<hr>", unsafe_allow_html=True)

            result = system_tech_prt.format(job_description=job_description, context=context)
            print("result", result)
            messages = [
                {"role": "system", "content": system_tech_prt.format(job_description=job_description, context=context)},
                {"role": "user", "content": f"Question: {query_text}"}
            ]

        elif interview_type == "Behavioral Interview":
            from config import system_behavioral_prt
            
            context = ""
            display_message(query_text, sender="user")
            st.markdown("<hr>", unsafe_allow_html=True)

            messages = [
                {"role": "system", "content": system_behavioral_prt.format(job_description=job_description)},
                {"role": "user", "content": f"Question: {query_text}"}
            ]

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
        st.info("💡 Try restarting the app or check your Streamlit installation.")
        return
        
    st.sidebar.title("🧭 Navigation")

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Validate API key
    if not api_key:
        st.error("❌ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.info("💡 Create a .env file in your project root with: OPENAI_API_KEY=your_api_key_here")
        return
    
    # Show API key status
    st.sidebar.success("✅ OpenAI API Key loaded from environment")
    
    # Add startup health check
    try:
        # Test basic imports
        import sys
        import gc
        st.sidebar.info(f"🐍 Python {sys.version.split()[0]}")
        st.sidebar.info(f"💾 Memory: {gc.get_count()}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ System check failed: {str(e)}")
    
    app_mode = st.sidebar.radio("Select App Mode:", options=["Interview", "Resume Builder"])

    # ChromaDB stats - only initialize when needed
    st.sidebar.title("ChromaDB Stats")
    if st.sidebar.button("🔄 Check ChromaDB Status"):
        try:
            client, collection, embedding_func = initialize_chromadb()
            if collection is not None:
                total_chunks = collection.count()
                st.sidebar.write(f"📊 Total Chunks: {total_chunks}")
            else:
                st.sidebar.warning("⚠️ ChromaDB not available")
        except Exception as e:
            st.sidebar.error(f"Error fetching chunk count: {str(e)}")
    else:
        st.sidebar.info("💡 Click 'Check ChromaDB Status' to initialize")

    # Show relevant content in main area
    if app_mode == "Interview":
        render_interview_ui(api_key)
    elif app_mode == "Resume Builder":
        render_resume_builder(api_key)

if __name__ == "__main__":
    main()