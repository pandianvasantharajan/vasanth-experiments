# Docker Deployment Guide for Model Service

This guide explains how to containerize and deploy the model service using Docker.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- API keys for Groq and OpenAI

### 1. Build and Run (Development)

```bash
# Navigate to model-service directory
cd model-service

# Make build script executable
chmod +x build.sh

# Build the Docker image
./build.sh

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and OPENAI_API_KEY

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

The service will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

### 2. Production Deployment

```bash
# Use production compose file with nginx and redis
docker-compose -f docker-compose.prod.yml up -d

# View all services
docker-compose -f docker-compose.prod.yml ps
```

## 📁 Files Overview

### Core Docker Files
- `Dockerfile` - Main container definition
- `docker-compose.yml` - Development setup
- `docker-compose.prod.yml` - Production setup with nginx and redis
- `.dockerignore` - Files to exclude from build context
- `requirements.txt` - Python dependencies
- `nginx.conf` - Nginx configuration for production

### Configuration Files
- `.env.example` - Environment variables template
- `build.sh` - Build and setup script

## 🔧 Configuration

### Environment Variables

```bash
# API Keys (required)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Service Configuration
PORT=8000
HOST=0.0.0.0

# Directories
MODEL_CACHE_DIR=/app/models
RESULTS_DIR=/app/results

# Logging
LOG_LEVEL=INFO
```

### Volume Mounts

- `../shared-models:/app/shared-models:ro` - Shared model files (read-only)
- `model-data:/app/models` - Model cache persistence
- `results-data:/app/results` - Results storage

## 🏗️ Architecture

### Development Setup
```
┌─────────────────┐
│   Docker Host   │
│                 │
│  ┌───────────┐  │
│  │Model      │  │
│  │Service    │  │
│  │:8000      │  │
│  └───────────┘  │
└─────────────────┘
```

### Production Setup
```
┌─────────────────────────────────────┐
│           Docker Host               │
│                                     │
│  ┌─────────┐  ┌───────────┐  ┌────┐ │
│  │ Nginx   │  │  Model    │  │Redis│ │
│  │ :80     │──│  Service  │  │     │ │
│  │ :443    │  │  :8000    │  │     │ │
│  └─────────┘  └───────────┘  └────┘ │
└─────────────────────────────────────┘
```

## 🐳 Docker Commands

### Build and Management
```bash
# Build image
docker build -t model-service:latest .

# Run container
docker run -d \
  --name model-service \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/../shared-models:/app/shared-models:ro \
  -v model-data:/app/models \
  -v results-data:/app/results \
  model-service:latest

# View logs
docker logs -f model-service

# Stop container
docker stop model-service

# Remove container
docker rm model-service
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up -d --build

# Scale service (if needed)
docker-compose up -d --scale model-service=3
```

## 🔍 Monitoring and Health Checks

### Health Check Endpoint
```bash
curl http://localhost:8000/health
```

### Container Health
```bash
# Check container health
docker ps

# View health check logs
docker inspect model-service | grep -A 10 Health
```

### Performance Monitoring
```bash
# View resource usage
docker stats model-service

# View container processes
docker exec model-service ps aux
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find and kill process using port 8000
   lsof -ti:8000 | xargs kill -9
   ```

2. **Missing API keys**
   ```bash
   # Check environment variables
   docker exec model-service env | grep API_KEY
   ```

3. **Volume mount issues**
   ```bash
   # Check if shared-models directory exists
   ls -la ../shared-models/
   ```

4. **Memory issues**
   ```bash
   # Check Docker memory settings
   docker system info | grep Memory
   ```

### Logs and Debugging
```bash
# View application logs
docker-compose logs model-service

# Debug container
docker exec -it model-service bash

# Check Python environment
docker exec model-service python -c "import sys; print(sys.path)"
```

## 🔒 Security Considerations

### Production Security
- Use secrets management for API keys
- Enable HTTPS with SSL certificates
- Implement proper firewall rules
- Regular security updates
- Monitor for vulnerabilities

### Network Security
```bash
# Create custom network
docker network create model-network --driver bridge

# Run with custom network
docker-compose up -d
```

## 📊 Performance Optimization

### Resource Limits
```yaml
# In docker-compose.yml
services:
  model-service:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Caching
- Redis integration for response caching
- Model file caching with persistent volumes
- Nginx caching for static content

## 🚀 Deployment Options

### Local Development
```bash
docker-compose up -d
```

### Production Server
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Cloud Deployment
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Kubernetes

## 📝 Maintenance

### Updates
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up -d --build

# Clean up old images
docker image prune -f
```

### Backup
```bash
# Backup volumes
docker run --rm -v model-data:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /data .
```

### Monitoring
- Set up log aggregation
- Configure alerting
- Monitor resource usage
- Track API performance metrics