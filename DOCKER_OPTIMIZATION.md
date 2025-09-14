# Docker Build Optimization Guide

This document explains the Docker caching optimizations implemented for the STS Neural Agent project.

## Problem

The original Dockerfile rebuilt everything from scratch on every build because:
- `COPY . .` invalidated all subsequent layers when any file changed
- CMake build happened after copying all source code
- No separation between frequently vs. infrequently changing files

## Solution: Layered Caching Strategy

### 1. **Dockerfile** (Standard - Optimized Layers)
- Separates C++ source from Python source
- C++ build layer caches unless C++ files change
- Python files copied last (most frequent changes)

**Performance:**
- First build: 10-15 minutes
- Python changes only: ~30 seconds
- C++ changes: 2-5 minutes

### 2. **Dockerfile.dev** (Development Optimized)
- Single-stage build optimized for development iteration
- Minimal layers for faster debugging
- Best for frequent Python code changes

**Performance:**
- First build: 8-12 minutes
- Subsequent builds: 30 seconds - 2 minutes

### 3. **Dockerfile.optimized** (Multi-Stage Production)
- Separate build stage for C++ compilation
- Minimal runtime image with only necessary files
- Best for production deployments

**Performance:**
- First build: 12-18 minutes
- Subsequent builds: 1-3 minutes
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

### Testing Different Variants

```bash
# Standard optimized build
./docker-scripts/test-docker.sh

# Development build (fastest Python iteration)
./docker-scripts/test-docker.sh --dev

# Multi-stage production build
./docker-scripts/test-docker.sh --optimized

# Force complete rebuild (when base image updates)
./docker-scripts/test-docker.sh --force-rebuild --no-cache
```

### Manual Building

```bash
# Development workflow
docker build -f Dockerfile.dev -t sts-dev .

# Production deployment
docker build -f Dockerfile.optimized -t sts-prod .

# Debug build issues
docker build --no-cache -t sts-debug .
```

## Cache Invalidation Scenarios

| Change Type | Cached Layers | Rebuild Time | Use Case |
|-------------|---------------|--------------|----------|
| Python code only | Base image → C++ build | ~30 seconds | Development |
| C++ code | Base image → Python deps | 2-5 minutes | Engine changes |
| Dependencies | Base image → System deps | 5-10 minutes | New packages |
| Base image | None | 10-15 minutes | Docker updates |

## Development Workflow Recommendations

1. **Daily Development**: Use `Dockerfile.dev` for Python code iteration
2. **C++ Changes**: Use standard `Dockerfile` with caching
3. **Production Builds**: Use `Dockerfile.optimized` for deployment
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
- Use `Dockerfile.optimized` for production
- Clean up build artifacts in final stage
- Consider using alpine-based images for smaller footprint

## Performance Monitoring

Track these metrics to optimize further:

- **Cache hit ratio**: Layers cached vs. total layers
- **Build time variance**: Consistency across builds
- **Image size growth**: Monitor final image bloat
- **Development velocity**: Time from code change to test

This optimization reduces typical development build times from 10+ minutes to under 1 minute while maintaining full functionality.