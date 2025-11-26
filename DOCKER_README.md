# VieNeu-TTS Docker Setup

This directory contains Docker configuration files to run VieNeu-TTS in containerized environments with support for both CPU and GPU execution.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- For GPU support: NVIDIA Docker runtime

### 1. CPU-only Setup (Recommended for testing)

```bash
# Using the convenience script
./docker-run.sh cpu

# Or using docker-compose directly
docker-compose --profile cpu up
```

Access the Gradio interface at: http://localhost:7860

### 2. GPU Setup (Requires NVIDIA Docker)

```bash
# Using the convenience script
./docker-run.sh gpu

# Or using docker-compose directly  
docker-compose --profile gpu up
```

Access the Gradio interface at: http://localhost:7861

### 3. Development Mode

```bash
# For development with live code reloading
./docker-run.sh dev

# Or using docker-compose directly
docker-compose --profile dev up
```

Access the development interface at: http://localhost:7862

## 📁 Directory Structure

The Docker setup creates and mounts the following directories:

```
VieNeu-TTS/
├── sample/           # Reference voice samples (mounted read-only)
├── utils/            # Text processing utilities (mounted read-only)  
├── output_audio/     # Generated audio files
├── models/           # Cached model files
├── cache/            # Hugging Face cache
├── config/           # Custom configuration files
├── custom_audio/     # User-uploaded audio files
└── input/            # Input files for batch processing
```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key configuration options:

- `DEVICE`: Set to `cpu` or `cuda`
- `BACKBONE_REPO`: Model repository (default: `pnnbao-ump/VieNeu-TTS`)
- `CODEC_REPO`: Codec repository (default: `neuphonic/neucodec`)
- `GRADIO_SERVER_PORT`: Web interface port
- `HF_TOKEN`: Hugging Face token (if needed for private models)

### Volume Mounts

The Docker setup automatically mounts:

- **Sample voices**: `./sample` → `/app/sample` (read-only)
- **Utils**: `./utils` → `/app/utils` (read-only)
- **Output**: `./output_audio` → `/app/output_audio`
- **Models**: `./models` → `/app/models`
- **Cache**: `./cache` → `/root/.cache/huggingface`
- **Config**: `./config` → `/app/config`
- **Custom audio**: `./custom_audio` → `/app/custom_audio`

## 🛠️ Available Services

### 1. CPU Service (`vieneu-tts-cpu`)
- **Port**: 7860
- **Profile**: `cpu`
- **Use case**: Testing, development, low-resource environments

### 2. GPU Service (`vieneu-tts-gpu`)  
- **Port**: 7861
- **Profile**: `gpu`
- **Requirements**: NVIDIA Docker runtime
- **Use case**: Production, faster inference

### 3. Development Service (`vieneu-tts-dev`)
- **Port**: 7862  
- **Profile**: `dev`
- **Features**: Live code mounting, development tools
- **Use case**: Code development and debugging

### 4. CLI Service (`vieneu-tts-cli`)
- **Profile**: `cli`
- **Use case**: Batch processing, automation

## 📋 Usage Examples

### Basic Usage

```bash
# Start CPU version
./docker-run.sh cpu

# Start GPU version (if available)
./docker-run.sh gpu

# Start in development mode
./docker-run.sh dev
```

### Advanced Usage

```bash
# Run on custom port
./docker-run.sh cpu --port 8080

# Run in background (detached mode)
./docker-run.sh gpu --detach

# Force rebuild images
./docker-run.sh cpu --build

# Run CLI for batch processing
./docker-run.sh cli
```

### Management Commands

```bash
# Build all images
./docker-run.sh build

# View logs
./docker-run.sh logs

# Stop all containers
./docker-run.sh stop

# Clean up everything
./docker-run.sh clean
```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   ./docker-run.sh cpu --port 8080
   ```

2. **GPU not detected**
   - Ensure NVIDIA Docker runtime is installed
   - Check: `docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi`

3. **Permission issues**
   ```bash
   sudo chown -R $USER:$USER output_audio/ models/ cache/
   ```

4. **Out of disk space**
   ```bash
   ./docker-run.sh clean
   docker system prune -a
   ```

### Checking System Requirements

```bash
# Check Docker version
docker --version
docker-compose --version

# Check NVIDIA Docker (for GPU support)
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check available disk space
df -h
```

## 🚀 Performance Tips

### CPU Optimization
- Set `TORCH_NUM_THREADS` and `OMP_NUM_THREADS` in `.env`
- Use smaller batch sizes for inference
- Consider using quantized models

### GPU Optimization  
- Ensure sufficient VRAM (8GB+ recommended)
- Use CUDA 11.8 compatible drivers
- Monitor GPU usage: `nvidia-smi`

### Storage Optimization
- Models are cached in `./models/` and `./cache/`
- Clean up old containers: `./docker-run.sh clean`
- Use `.dockerignore` to exclude unnecessary files

## 🔒 Security Considerations

- The Gradio interface binds to `0.0.0.0` inside containers
- Use reverse proxy (nginx) for production deployments
- Keep Hugging Face tokens secure in `.env` files
- Regularly update base images for security patches

## 📚 Additional Resources

- [VieNeu-TTS GitHub Repository](https://github.com/pnnbao97/VieNeu-TTS)
- [Hugging Face Model Card](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)

## 🤝 Contributing

When contributing Docker-related changes:

1. Test both CPU and GPU configurations
2. Update this README if adding new features
3. Ensure `.dockerignore` excludes unnecessary files
4. Test the convenience script on different platforms

## 📄 License

This Docker configuration follows the same license as the main VieNeu-TTS project (Apache License 2.0).
