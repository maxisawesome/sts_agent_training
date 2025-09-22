# Docker Build Optimization Guide

This document explains the Docker caching optimizations implemented for the STS Neural Agent project.

## Problem

The original Dockerfile rebuilt everything from scratch on every build because:
- `COPY . .` invalidated all subsequent layers when any file changed
- CMake build happened after copying all source code
- No separation between frequently vs. infrequently changing files

## Solution: Layered Caching Strategy

### **Dockerfile** (Optimized Layers)
- Separates C++ source from Python source
- C++ build layer caches unless C++ files change
- Python files copied last (most frequent changes)
- Uses multi-stage build for optimal layer caching

**Performance:**
- First build: 10-15 minutes
- Python changes only: ~30 seconds
- C++ changes: 2-5 minutes
- Dependency changes: 5-10 minutes
- Smallest final image size

## Cache Efficiency Layers

From most stable (cached longest) to most volatile:

1. **Base Image** (`FROM pytorch/pytorch:latest`)
2. **System Dependencies** (`apt-get install build-essential cmake...`)
3. **Python Dependencies** (`pip install -r requirements.txt`)
4. **C++ Source Code** (`sts_lightspeed/` directories)
5. **C++ Build** (`cmake && make`)
6. **Python Source Code** (`*.py` files)

## Usage

### Testing

```bash
# Optimized build with caching
./docker-scripts/test-docker.sh

# Build-only test (no functionality tests)
./docker-scripts/test-docker.sh --build-only

# Force complete rebuild (when base image updates)
./docker-scripts/test-docker.sh --force-rebuild --no-cache
```

### Manual Building

```bash
# Standard optimized build
docker build -t sts-neural-agent .

# Debug build issues (no cache)
docker build --no-cache -t sts-neural-agent .

# Build with specific tag
docker build -t sts-neural-agent:v1.0 .
```

## Cache Invalidation Scenarios

| Change Type | Cached Layers | Rebuild Time | Use Case |
|-------------|---------------|--------------|----------|
| Python code only | Base image → C++ build | ~30 seconds | Development |
| C++ code | Base image → Python deps | 2-5 minutes | Engine changes |
| Dependencies | Base image → System deps | 5-10 minutes | New packages |
| Base image | None | 10-15 minutes | Docker updates |

## Development Workflow Recommendations

1. **Daily Development**: Use standard `Dockerfile` with caching for all development
2. **C++ Changes**: Rebuild will cache all layers up to C++ source changes
3. **Production Builds**: Same `Dockerfile` works for production deployment
4. **CI/CD**: Use `--no-cache` periodically to ensure clean builds

## Monitoring Cache Efficiency

The test script shows cache statistics:
```bash
🚀 Cache efficiency: 8 layers cached
```

Good cache efficiency indicators:
- 5+ cached layers for Python-only changes
- 3+ cached layers for C++ changes
- Build time under 2 minutes for typical changes

## Troubleshooting

### Slow Builds Despite Caching
- Check if `.dockerignore` is properly configured
- Verify file timestamps aren't changing unnecessarily
- Use `docker system df` to check cache usage

### Cache Not Working
- Ensure you're not using `--no-cache` unintentionally
- Check that base image hasn't changed
- Verify Docker daemon has sufficient disk space

### Large Image Sizes
- The main `Dockerfile` uses multi-stage builds for optimized production images
- Build artifacts are cleaned up in the final stage
- Consider using alpine-based images for even smaller footprint

## Performance Monitoring

Track these metrics to optimize further:

- **Cache hit ratio**: Layers cached vs. total layers
- **Build time variance**: Consistency across builds
- **Image size growth**: Monitor final image bloat
- **Development velocity**: Time from code change to test

This optimization reduces typical development build times from 10+ minutes to under 1 minute while maintaining full functionality.