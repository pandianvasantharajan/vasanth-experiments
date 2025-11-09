#!/bin/bash

# Quick Docker deployment script
set -e

echo "🐳 Starting Docker-based Model Service Deployment"
echo "================================================="

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker daemon is not running."
        echo ""
        echo "To start Docker:"
        echo "1. Open Docker Desktop application"
        echo "2. Wait for Docker to start (whale icon appears in menu bar)"
        echo "3. Run this script again"
        echo ""
        echo "Alternatively, start Docker from command line:"
        echo "   open -a Docker"
        exit 1
    fi
    echo "✅ Docker is running"
}

# Build Docker image
build_image() {
    echo "📦 Building Docker image..."
    docker build -t model-service:latest .
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully!"
    else
        echo "❌ Failed to build Docker image"
        exit 1
    fi
}

# Run container
run_container() {
    echo "🚀 Starting Docker container..."
    
    # Stop any existing container
    docker stop model-service 2>/dev/null || true
    docker rm model-service 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name model-service \
        -p 8000:8000 \
        --env-file .env \
        -v "$(pwd)/../shared-models:/app/shared-models:ro" \
        -v model-data:/app/models \
        -v results-data:/app/results \
        model-service:latest
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started successfully!"
        echo ""
        echo "🌐 Service URLs:"
        echo "   API: http://localhost:8000"
        echo "   Swagger UI: http://localhost:8000/docs"
        echo "   Health Check: http://localhost:8000/health"
        echo ""
        echo "📊 To view logs:"
        echo "   docker logs -f model-service"
        echo ""
        echo "🛑 To stop the service:"
        echo "   docker stop model-service"
    else
        echo "❌ Failed to start container"
        exit 1
    fi
}

# Check container status
check_status() {
    echo "📊 Checking container status..."
    sleep 5
    
    if docker ps | grep -q model-service; then
        echo "✅ Container is running"
        
        # Test health endpoint
        echo "🔍 Testing health endpoint..."
        sleep 10
        
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ Service is healthy and responding"
            echo ""
            echo "🎉 Docker deployment completed successfully!"
            echo "   You can now access http://localhost:8000/docs"
        else
            echo "⚠️ Container is running but service may still be starting..."
            echo "   Check logs: docker logs -f model-service"
        fi
    else
        echo "❌ Container failed to start"
        echo "   Check logs: docker logs model-service"
    fi
}

# Main execution
main() {
    check_docker
    build_image
    run_container
    check_status
}

# Run main function
main