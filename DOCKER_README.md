# Docker Deployment Guide

This guide explains how to deploy the Interview Bot application using Docker.

## Prerequisites

- Docker installed on your system ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (usually included with Docker Desktop)
- OpenAI API key

## Quick Start

### 1. Set up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Build and Run with Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The application will be available at: `http://localhost:8501`

### 3. Build and Run with Docker (Manual)

```bash
# Build the Docker image
docker build -t interview-bot .

# Run the container
docker run -d \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  -v $(pwd)/chroma_data:/app/chroma_data \
  -v $(pwd)/temp_pdfs:/app/temp_pdfs \
  --name interview-bot \
  interview-bot

# View logs
docker logs -f interview-bot

# Stop the container
docker stop interview-bot
docker rm interview-bot
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered
- `CUDA_VISIBLE_DEVICES=""`: Disables GPU (set to empty string)
- `TOKENIZERS_PARALLELISM=false`: Prevents tokenizer warnings

### Volumes

The Docker setup mounts the following directories:

- `./chroma_data:/app/chroma_data` - Persists ChromaDB vector database
- `./temp_pdfs:/app/temp_pdfs` - Persists temporary PDF files

### Ports

- `8501` - Streamlit default port

## Troubleshooting

### Container won't start

1. Check logs: `docker-compose logs` or `docker logs interview-bot`
2. Verify environment variables are set correctly
3. Ensure port 8501 is not already in use

### Typst not found

The Dockerfile installs Typst automatically. If you see errors:
- Check that the Typst installation step completed successfully
- Verify PATH includes `/root/.typst/bin`

### ChromaDB issues

- Ensure the `chroma_data` directory exists and has proper permissions
- Check that the volume mount is working: `docker exec interview-bot ls -la /app/chroma_data`

### PDF generation fails

- Verify Typst is installed: `docker exec interview-bot typst --version`
- Check temp_pdfs directory permissions
- Review application logs for specific error messages

## Production Deployment

For production deployment, consider:

1. **Use environment secrets**: Don't hardcode API keys
2. **Add reverse proxy**: Use Nginx or Traefik in front of Streamlit
3. **Enable HTTPS**: Configure SSL certificates
4. **Resource limits**: Set memory and CPU limits in docker-compose.yml
5. **Monitoring**: Add health checks and logging
6. **Backup**: Regularly backup the `chroma_data` volume

### Example Production docker-compose.yml additions:

```yaml
services:
  interview-bot:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Development

### Rebuild after code changes

```bash
docker-compose up -d --build
```

### Access container shell

```bash
docker exec -it interview-bot /bin/bash
```

### View real-time logs

```bash
docker-compose logs -f interview-bot
```

## Clean Up

Remove containers, volumes, and images:

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes ChromaDB data)
docker-compose down -v

# Remove images
docker rmi interview-bot
```

