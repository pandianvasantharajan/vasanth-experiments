# Docker Deployment Guide

This guide explains how to deploy the Model Service and Model Dashboard using Docker.

## Services

1. **model-service** - FastAPI backend serving ML models (port 8000)
   - YouTube Video Analyzer (RAG pipeline)
   - Temperature Forecasting
   - Object Detection

2. **model-dashboard** - React frontend (port 3000)
   - User interface for interacting with models
   - Proxy API requests to model-service

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- API Keys (optional, for YouTube analyzer):
  - GROQ_API_KEY
  - OPENAI_API_KEY
  - OPENROUTER_API_KEY

## Quick Start

### 1. Set up environment variables

Create or copy the `.env` file:

```bash
# From youtube-video-analyser-model directory
cp youtube-video-analyser-model/.env.example youtube-video-analyser-model/.env

# Edit with your API keys
nano youtube-video-analyser-model/.env
```

### 2. Build and run services

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up --build -d
```

### 3. Access the services

- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/

## Docker Commands

### Build services
```bash
docker-compose build
```

### Start services
```bash
docker-compose up
```

### Stop services
```bash
docker-compose down
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f model-service
docker-compose logs -f model-dashboard
```

### Restart a service
```bash
docker-compose restart model-service
```

### Remove everything (including volumes)
```bash
docker-compose down -v
```

## Environment Variables

Set these in `youtube-video-analyser-model/.env` or pass via docker-compose:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `LLM_PROVIDER` | LLM provider to use | `groq` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNKING_METHOD` | Text chunking method | `langchain` |
| `CHUNK_SIZE` | Chunk size for text splitting | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

## API Endpoints

### YouTube Analyzer
```bash
# Analyze a YouTube video
curl -X POST http://localhost:8000/analyze/youtube \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID"
  }'
```

### Temperature Forecasting
```bash
# Predict temperature
curl -X POST http://localhost:8000/predict/temperature \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 14,
    "month": 7,
    "day_of_year": 195
  }'
```

### Object Detection
```bash
# Detect objects in an image
curl -X POST http://localhost:8000/detect/objects \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5"
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Models Info
```bash
curl http://localhost:8000/models
```

## Troubleshooting

### Port already in use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Changed from 8000:8000
```

### Out of memory
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or add memory limits to docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G
```

### Container won't start
```bash
# Check logs
docker-compose logs model-service

# Rebuild without cache
docker-compose build --no-cache

# Remove old containers and volumes
docker-compose down -v
docker system prune -a
```

### API returns 503
- Check if YouTube analyzer pickle file exists
- Verify environment variables are set
- Check model-service logs for errors

## Development

### Local development with hot reload
```bash
# Mount local code as volume (edit docker-compose.yml)
volumes:
  - ./model-service:/app
  - ./model-service/youtube_analyzer.pkl:/app/youtube_analyzer.pkl

# Restart on code changes
docker-compose restart model-service
```

### Debugging inside container
```bash
# Open shell in running container
docker exec -it model-service bash

# Run Python commands
python -c "import main; print(main.model_service.youtube_analyzer)"
```

## Production Deployment

For production, consider:

1. **Use environment-specific compose files**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
   ```

2. **Enable HTTPS** with reverse proxy (nginx/traefik)

3. **Set resource limits** in docker-compose.yml

4. **Use secrets management** (Docker secrets, Vault)

5. **Add monitoring** (Prometheus, Grafana)

6. **Set up logging** (ELK stack, Loki)

7. **Configure backups** for ChromaDB data volume

## Network Architecture

```
Client (Browser)
    ↓ Port 3000
[model-dashboard (nginx)]
    ↓ /api/* → model-service:8000
[model-service (FastAPI)]
    ↓
[ChromaDB (persistent volume)]
```

## Volumes

- `./data/youtube_chroma_db` - ChromaDB vector database persistence
- Survives container restarts
- Backed up automatically if using volume mounts

## Support

For issues, check:
1. Docker logs: `docker-compose logs`
2. Container status: `docker-compose ps`
3. Network connectivity: `docker network inspect vasanth-experiments_model-network`
