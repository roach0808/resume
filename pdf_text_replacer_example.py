"""
Example usage of PDF Text Replacer
Demonstrates how to replace text in PDF while preserving style
"""

from pdf_text_replacer import PDFTextReplacer, replace_text_in_pdf


def example_simple_replace():
    """Simple example: Replace text and save to new file"""
    
    # Replace "xxxx" with "yyyy" in the PDF
    replace_text_in_pdf(
        pdf_path="resume.pdf",
        old_text="xxxx",
        new_text="yyyy",
        output_path="resume_updated.pdf",
        case_sensitive=False
    )
    
    print("✅ Text replaced and saved to resume_updated.pdf")


def example_replace_job_description():
    """Example: Replace job description text"""
    
    # Find and replace job description
    pdf_bytes = replace_text_in_pdf(
        pdf_path="resume.pdf",
        old_text="Software Engineer",
        new_text="Senior Machine Learning Engineer",
        case_sensitive=False
    )
    
    # Save bytes to file if needed
    if pdf_bytes:
        with open("resume_updated.pdf", "wb") as f:
            f.write(pdf_bytes)
    
    print("✅ Job description updated")


def example_advanced_usage():
    """Advanced example with more control"""
    
    # Create replacer instance
    replacer = PDFTextReplacer("resume.pdf")
    
    # Find all instances of text with their styles
    instances = replacer.find_text_instances("xxxx")
    print(f"Found {len(instances)} instances of 'xxxx'")
    
    for i, instance in enumerate(instances):
        print(f"Instance {i+1}:")
        print(f"  Page: {instance['page']}")
        print(f"  Style: {instance['style']}")
        print(f"  Position: {instance['bbox']}")
    
    # Replace text with style preservation
    count = replacer.replace_text_advanced(
        old_text="xxxx",
        new_text="yyyy",
        case_sensitive=False,
        preserve_formatting=True
    )
    
    print(f"\nReplaced {count} instance(s)")
    
    # Save to file
    replacer.save("resume_updated.pdf")
    
    # Close the document
    replacer.close()
    
    print("✅ PDF saved with replacements")


def example_streamlit_integration():
    """Example for Streamlit integration"""
    
    import streamlit as st
    from pdf_text_replacer import replace_text_in_pdf
    
    st.title("PDF Text Replacer")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    old_text = st.text_input("Text to find", "xxxx")
    new_text = st.text_input("Replacement text", "yyyy")
    
    if st.button("Replace Text") and uploaded_file:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Replace text
            pdf_bytes = replace_text_in_pdf(
                pdf_path=tmp_path,
                old_text=old_text,
                new_text=new_text
            )
            
            # Download button
            st.download_button(
                label="📥 Download Updated PDF",
                data=pdf_bytes,
                file_name="updated.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            import os
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("PDF Text Replacer Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # example_simple_replace()
    # example_replace_job_description()
    # example_advanced_usage()
    
    print("\nNote: Update the file paths in the examples before running.")

