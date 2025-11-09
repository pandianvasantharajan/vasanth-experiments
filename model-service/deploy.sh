#!/bin/bash

# Complete deployment script for model-service

set -e

echo "🚀 Model Service Docker Deployment"
echo "=================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker Desktop and try again."
        echo ""
        echo "To start Docker:"
        echo "1. Open Docker Desktop application"
        echo "2. Wait for Docker to start completely"
        echo "3. Run this script again"
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to create environment file
setup_env() {
    if [ ! -f .env ]; then
        echo "📋 Creating .env file from template..."
        cp .env.example .env
        echo "⚠️  IMPORTANT: Please edit .env file and add your API keys:"
        echo "   - GROQ_API_KEY"
        echo "   - OPENAI_API_KEY"
        echo ""
        echo "You can get these keys from:"
        echo "- Groq: https://console.groq.com/keys"
        echo "- OpenAI: https://platform.openai.com/api-keys"
        echo ""
        read -p "Press Enter after you've added your API keys to .env file..."
    else
        echo "✅ .env file already exists"
    fi
}

# Function to build Docker image
build_image() {
    echo "📦 Building Docker image..."
    docker build -t model-service:latest .
    echo "✅ Docker image built successfully!"
}

# Function to run with Docker Compose
run_compose() {
    echo "🚀 Starting services with Docker Compose..."
    docker-compose up -d
    
    echo ""
    echo "✅ Services started successfully!"
    echo ""
    echo "🌐 Service URLs:"
    echo "   API: http://localhost:8000"
    echo "   Swagger UI: http://localhost:8000/docs"
    echo "   Health Check: http://localhost:8000/health"
    echo ""
    echo "📊 To view logs:"
    echo "   docker-compose logs -f"
    echo ""
    echo "🛑 To stop services:"
    echo "   docker-compose down"
}

# Function to run production setup
run_production() {
    echo "🏭 Starting production setup..."
    docker-compose -f docker-compose.prod.yml up -d
    
    echo ""
    echo "✅ Production services started!"
    echo ""
    echo "🌐 Service URLs:"
    echo "   Main: http://localhost (via Nginx)"
    echo "   Direct API: http://localhost:8000"
    echo "   Swagger UI: http://localhost:8000/docs"
    echo ""
    echo "📊 Production services:"
    echo "   - Model Service (FastAPI)"
    echo "   - Nginx (Load Balancer/Proxy)"
    echo "   - Redis (Caching)"
}

# Main execution
main() {
    echo "Checking prerequisites..."
    check_docker
    
    echo ""
    echo "Choose deployment option:"
    echo "1) Development (Docker Compose)"
    echo "2) Production (with Nginx + Redis)"
    echo "3) Build only"
    echo ""
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            setup_env
            build_image
            run_compose
            ;;
        2)
            setup_env
            build_image
            run_production
            ;;
        3)
            build_image
            echo ""
            echo "🐳 Image built successfully!"
            echo "To run manually:"
            echo "  docker run -d --name model-service -p 8000:8000 --env-file .env model-service:latest"
            ;;
        *)
            echo "❌ Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
}

# Run main function
main