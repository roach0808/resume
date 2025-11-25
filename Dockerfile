# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - curl and wget for downloading Typst
# - build-essential for compiling some Python packages
# - fonts for PDF generation
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install Typst (required for RenderCV)
# Download and install Typst binary directly
RUN curl -L "https://github.com/typst/typst/releases/latest/download/typst-x86_64-unknown-linux-musl.tar.xz" -o /tmp/typst.tar.xz && \
    tar -xf /tmp/typst.tar.xz -C /tmp && \
    mv /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst && \
    chmod +x /usr/local/bin/typst && \
    rm -rf /tmp/typst* && \
    typst --version

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be missing
RUN pip install --no-cache-dir \
    rendercv \
    reportlab

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p temp_pdfs chroma_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV TOKENIZERS_PARALLELISM="false"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

