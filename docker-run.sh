#!/bin/bash

# VieNeu-TTS Docker Run Script
# This script provides easy commands to run VieNeu-TTS with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if NVIDIA Docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_warning "NVIDIA Docker runtime not available. GPU mode will not work."
        return 1
    fi
    return 0
}

# Function to create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p output_audio models cache config custom_audio input
    print_success "Directories created successfully"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  cpu           Run VieNeu-TTS with CPU support"
    echo "  gpu           Run VieNeu-TTS with GPU support (requires NVIDIA Docker)"
    echo "  dev           Run VieNeu-TTS in development mode"
    echo "  cli           Run VieNeu-TTS CLI for batch processing"
    echo "  build         Build Docker images"
    echo "  clean         Clean up Docker images and containers"
    echo "  logs          Show logs from running containers"
    echo "  stop          Stop all running containers"
    echo ""
    echo "Options:"
    echo "  --port PORT   Specify port (default: 7860 for CPU, 7861 for GPU)"
    echo "  --detach      Run in detached mode"
    echo "  --build       Force rebuild of images"
    echo ""
    echo "Examples:"
    echo "  $0 cpu                    # Run CPU version"
    echo "  $0 gpu --port 8080        # Run GPU version on port 8080"
    echo "  $0 dev --detach           # Run development version in background"
    echo "  $0 build                  # Build all images"
}

# Parse command line arguments
COMMAND=""
PORT=""
DETACH=""
BUILD_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        cpu|gpu|dev|cli|build|clean|logs|stop)
            COMMAND="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --detach)
            DETACH="-d"
            shift
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [[ -z "$COMMAND" ]]; then
    print_error "No command provided"
    show_usage
    exit 1
fi

# Check Docker
check_docker

# Create directories
create_directories

# Execute commands
case $COMMAND in
    cpu)
        print_info "Starting VieNeu-TTS with CPU support..."
        PORT=${PORT:-7860}
        docker-compose --profile cpu up $DETACH $BUILD_FLAG
        if [[ -z "$DETACH" ]]; then
            print_success "VieNeu-TTS CPU is running on http://localhost:$PORT"
        fi
        ;;
    
    gpu)
        print_info "Checking NVIDIA Docker support..."
        if check_nvidia_docker; then
            print_info "Starting VieNeu-TTS with GPU support..."
            PORT=${PORT:-7861}
            docker-compose --profile gpu up $DETACH $BUILD_FLAG
            if [[ -z "$DETACH" ]]; then
                print_success "VieNeu-TTS GPU is running on http://localhost:$PORT"
            fi
        else
            print_error "GPU support not available. Please install NVIDIA Docker runtime."
            exit 1
        fi
        ;;
    
    dev)
        print_info "Starting VieNeu-TTS in development mode..."
        PORT=${PORT:-7862}
        docker-compose --profile dev up $DETACH $BUILD_FLAG
        if [[ -z "$DETACH" ]]; then
            print_success "VieNeu-TTS Development is running on http://localhost:$PORT"
        fi
        ;;
    
    cli)
        print_info "Running VieNeu-TTS CLI..."
        docker-compose --profile cli up $BUILD_FLAG
        ;;
    
    build)
        print_info "Building Docker images..."
        docker-compose build
        print_success "Docker images built successfully"
        ;;
    
    clean)
        print_info "Cleaning up Docker images and containers..."
        docker-compose down --rmi all --volumes --remove-orphans
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    
    logs)
        print_info "Showing logs from running containers..."
        docker-compose logs -f
        ;;
    
    stop)
        print_info "Stopping all containers..."
        docker-compose down
        print_success "All containers stopped"
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
