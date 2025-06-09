# CUDA Build Guide for Torch-PanguLU

This document provides a comprehensive guide for building Torch-PanguLU with CUDA support.

## 🚀 Quick Start

### Automated Setup (Recommended)

We provide a fully automated setup script that handles all configuration:

```bash
# CPU-only build (recommended - stable)
bash scripts/setup_pangulu.sh --cpu-only

# Experimental CUDA build
bash scripts/setup_pangulu.sh --enable-cuda

# View all options
bash scripts/setup_pangulu.sh --help
```

### Build Status

| Build Type | Status | Description |
|------------|--------|-------------|
| **CPU-Only** | ✅ **Stable** | Production-ready with excellent numerical accuracy (1e-15 residuals) |
| **CUDA** | ⚠️ **Experimental** | Known compatibility issues with PanguLU CUDA kernels |

## 📋 CUDA Prerequisites

- **GPU**: NVIDIA GPU with Compute Capability 6.0+ (GTX 10 series or newer)
- **CUDA Toolkit**: Version 11.0+ (CUDA 12.x recommended)
- **Drivers**: Compatible NVIDIA driver (470.x+ for CUDA 11, 520.x+ for CUDA 12)
- **System**: Linux (Ubuntu/Debian/CentOS recommended)

### Check Your System

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Check PyTorch CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🔧 Manual CUDA Configuration

If you need fine-grained control over the build process:

### Step 1: Configure PanguLU for CUDA

Edit `third_party/PanguLU/make.inc`:

```make
# Basic configuration
COMPILE_LEVEL = -O3
CC = gcc $(COMPILE_LEVEL)
MPICC = mpicc $(COMPILE_LEVEL)
OPENBLAS_INC = -I/usr/include/x86_64-linux-gnu/openblas-pthread/
OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas

# CUDA configuration (adjust paths as needed)
CUDA_PATH = /usr/local/cuda  # or /usr/lib/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse

# NVCC compiler settings
NVCC = nvcc $(COMPILE_LEVEL)
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg \
            -gencode=arch=compute_60,code=sm_60 \
            -gencode=arch=compute_70,code=sm_70 \
            -gencode=arch=compute_80,code=sm_80 \
            $(CUDA_INC)

# Linking configuration
MPICCFLAGS = $(OPENBLAS_INC) $(CUDA_INC) $(OPENBLAS_LIB) $(CUDA_LIB) -fopenmp -lpthread -lm
MPICCLINK = $(OPENBLAS_LIB) $(CUDA_LIB)

# Enable GPU support (this is the key flag)
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN
```

### Step 2: Build PanguLU

```bash
cd third_party/PanguLU
make clean
make -C src -j$(nproc)  # Build source files
make -C lib             # Create libraries
```

### Step 3: Configure Torch Extension

```bash
# Force CUDA support in setup.py
export TORCH_CUDA_FORCE=1

# Build the extension
python setup.py build_ext --inplace
```

### Step 4: Verify CUDA Build

```bash
python -c "
import torch_pangulu
info = torch_pangulu._C.get_pangulu_info()
print('CUDA support:', info['cuda_support'])
print('Available:', info['available'])
"
```

## ⚠️ Known CUDA Issues

### 1. Compilation Errors

**Issue**: `atomicAdd` errors with double precision
```
error: no instance of overloaded function "atomicAdd" matches the argument list
        argument types are: (double *, double)
```

**Cause**: PanguLU's CUDA kernels use `atomicAdd` with double precision, which requires Compute Capability 6.0+

**Solution**: Use CPU-only build or patch CUDA kernels for newer GPU architectures

### 2. Kernel Compatibility Issues

**Issue**: CUDA kernels fail to compile with modern CUDA versions

**Solution**: 
```bash
# Fall back to stable CPU build
bash scripts/setup_pangulu.sh --cpu-only --force-rebuild
```

### 3. Memory Requirements

**Issue**: CUDA version requires substantial GPU memory

**Solution**: Use CPU version for large matrices or reduce problem size

## 🛠️ Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation paths
ls /usr/local/cuda/bin/nvcc  # Standard install
ls /usr/lib/cuda/bin/nvcc    # Ubuntu package

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

### Driver/Runtime Mismatch

```bash
# Check versions
nvcc --version                    # Toolkit version
nvidia-smi | grep "CUDA Version"  # Driver version

# Update drivers if needed
sudo apt update && sudo apt install nvidia-driver-535  # Ubuntu
```

### Build Failures

```bash
# Clean and retry
bash scripts/setup_pangulu.sh --cpu-only --force-rebuild --verbose

# Check build logs
tail -50 third_party/PanguLU/build.log
tail -50 torch_build.log
```

## 🎯 Recommendations

1. **Start with CPU-only**: The CPU version is stable, fast, and has excellent numerical accuracy
2. **Test CUDA only if needed**: Only attempt CUDA if you have large matrices and compatible hardware
3. **Use automated script**: The `setup_pangulu.sh` script handles most configuration automatically
4. **Check compatibility first**: Verify your GPU compute capability before attempting CUDA build

## 📊 Performance Comparison

| Build Type | Matrix Size | Performance | Stability | Memory Usage |
|------------|-------------|-------------|-----------|--------------|
| CPU-Only | All sizes | Excellent | ✅ Stable | Low |
| CUDA | Large (>100k×100k) | Potentially faster | ⚠️ Experimental | High GPU memory |

## 🔗 Additional Resources

- **[BUILD.md](BUILD.md)**: Comprehensive build instructions for all platforms
- **[README.md](README.md)**: General project documentation and examples
- **PanguLU Documentation**: https://github.com/SuperScientificSoftwareLaboratory/PanguLU
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit

---

**Note**: The automated script `scripts/setup_pangulu.sh` is the recommended approach for most users. It handles dependency checking, environment setup, building, and testing automatically.