#!/bin/bash

# Docker Management Script for Model Service and Dashboard
# Usage: ./docker-run.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker and docker-compose are installed
check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_info "Docker and Docker Compose are installed ✓"
}

# Build all services
build() {
    print_info "Building Docker images..."
    docker-compose build "$@"
    print_info "Build completed successfully ✓"
}

# Start services
start() {
    print_info "Starting services..."
    docker-compose up -d
    print_info "Services started ✓"
    print_info "Dashboard: http://localhost:3000"
    print_info "API: http://localhost:8000"
    print_info "API Docs: http://localhost:8000/docs"
}

# Stop services
stop() {
    print_info "Stopping services..."
    docker-compose down
    print_info "Services stopped ✓"
}

# Restart services
restart() {
    print_info "Restarting services..."
    docker-compose restart "$@"
    print_info "Services restarted ✓"
}

# View logs
logs() {
    if [ -z "$1" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$1"
    fi
}

# Check status
status() {
    print_info "Service Status:"
    docker-compose ps
    echo ""
    print_info "Network Status:"
    docker network inspect vasanth-experiments_model-network 2>/dev/null | grep -A 5 "Containers" || print_warn "Network not found"
}

# Clean up everything
clean() {
    print_warn "This will remove all containers, volumes, and images. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Cleaning up..."
        docker-compose down -v
        docker system prune -f
        print_info "Cleanup completed ✓"
    else
        print_info "Cleanup cancelled"
    fi
}

# Run tests
test() {
    print_info "Testing API endpoints..."
    
    # Check if services are running
    if ! docker-compose ps | grep -q "Up"; then
        print_error "Services are not running. Start them first with: ./docker-run.sh start"
        exit 1
    fi
    
    echo ""
    print_info "Testing health endpoint..."
    curl -f http://localhost:8000/ || print_error "Health check failed"
    
    echo ""
    print_info "Testing models info..."
    curl -f http://localhost:8000/models || print_error "Models info failed"
    
    echo ""
    print_info "Testing dashboard..."
    curl -f http://localhost:3000/ > /dev/null || print_error "Dashboard failed"
    
    echo ""
    print_info "All tests passed ✓"
}

# Show help
help() {
    cat << EOF
Docker Management Script for Model Service and Dashboard

Usage: $0 [command] [options]

Commands:
    build           Build all Docker images
    start           Start all services in detached mode
    stop            Stop all services
    restart [svc]   Restart all services or specific service
    logs [svc]      View logs (all services or specific service)
    status          Show service and network status
    test            Run basic API tests
    clean           Remove all containers, volumes, and images
    help            Show this help message

Examples:
    $0 build                    # Build all images
    $0 build --no-cache         # Build without cache
    $0 start                    # Start services
    $0 logs model-service       # View model-service logs
    $0 restart model-dashboard  # Restart dashboard
    $0 test                     # Test API endpoints
    $0 clean                    # Clean up everything

Services:
    - model-service: FastAPI backend (port 8000)
    - model-dashboard: React frontend (port 3000)

EOF
}

# Main script
main() {
    check_requirements
    
    case "${1:-help}" in
        build)
            shift
            build "$@"
            ;;
        start)
            start
            ;;
        stop)
            stop
            ;;
        restart)
            shift
            restart "$@"
            ;;
        logs)
            shift
            logs "$@"
            ;;
        status)
            status
            ;;
        test)
            test
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            help
            exit 1
            ;;
    esac
}

main "$@"
