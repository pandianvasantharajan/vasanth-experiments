#!/bin/bash

# Build and run script for model-service Docker container

set -e

echo "🐳 Building model-service Docker container..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t model-service:latest .

echo "✅ Docker image built successfully!"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys before running the container."
    echo "   Required: GROQ_API_KEY, OPENAI_API_KEY"
fi

echo "🚀 To run the container, use one of these commands:"
echo ""
echo "Using Docker Compose (recommended):"
echo "  docker-compose up -d"
echo ""
echo "Using Docker run:"
echo "  docker run -d \\"
echo "    --name model-service \\"
echo "    -p 8000:8000 \\"
echo "    --env-file .env \\"
echo "    -v \$(pwd)/../shared-models:/app/shared-models:ro \\"
echo "    -v model-data:/app/models \\"
echo "    -v results-data:/app/results \\"
echo "    model-service:latest"
echo ""
echo "🌐 Once running, the service will be available at:"
echo "   http://localhost:8000"
echo "   Swagger UI: http://localhost:8000/docs"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f"
echo "   # or"
echo "   docker logs -f model-service"
echo ""
echo "🛑 To stop the service:"
echo "   docker-compose down"
echo "   # or"
echo "   docker stop model-service"