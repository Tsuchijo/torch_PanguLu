# Build Instructions

This document provides comprehensive instructions for building the Torch-PanguLU project on different platforms.

> **📋 Quick Start**: For the latest simplified build instructions with working PanguLU integration, see the **[README.md](README.md)** installation section. This document contains more detailed platform-specific information.

> **✅ Status**: PanguLU integration is now complete and working with excellent numerical accuracy!

## Prerequisites

### System Requirements

- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **Architecture**: x86_64 or ARM64
- **GPU**: NVIDIA GPU with CUDA 11.0+ (optional but recommended for performance)
- **Memory**: At least 8GB RAM (16GB+ recommended for large matrices)

### Software Dependencies

#### Required Dependencies

1. **C++ Compiler**: GCC 7.0+ or Clang 10.0+
2. **CMake**: Version 3.12 or later
3. **Python**: Version 3.7 or later
4. **PyTorch**: Version 1.9.0 or later
5. **MPI**: OpenMPI 4.0+ or Intel MPI 2019+
6. **BLAS Library**: OpenBLAS, Intel MKL, or ATLAS

#### Optional Dependencies

1. **CUDA Toolkit**: Version 11.0+ (for GPU acceleration) - **See CUDA Build Section below**
2. **OpenMP**: For CPU parallelization
3. **METIS**: For matrix reordering (performance optimization)

## Step-by-Step Build Process

### 1. Environment Setup

#### On Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install build essentials
sudo apt install build-essential cmake git

# Install MPI
sudo apt install openmpi-bin openmpi-common libopenmpi-dev

# Install BLAS
sudo apt install libopenblas-dev

# Install Python dependencies
pip install torch torchvision numpy pybind11 scipy

# Optional: CUDA (if not already installed)
# Follow NVIDIA's official CUDA installation guide
```

#### On CentOS/RHEL/Fedora:
```bash
# Install build tools
sudo yum groupinstall "Development Tools"
sudo yum install cmake git

# Install MPI
sudo yum install openmpi openmpi-devel
# Add MPI to PATH (add to ~/.bashrc)
export PATH=$PATH:/usr/lib64/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib

# Install BLAS
sudo yum install openblas-devel

# Install Python dependencies
pip install torch torchvision numpy pybind11 scipy
```

#### On macOS:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install open-mpi openblas libomp cmake git

# Install Python dependencies
pip install torch torchvision numpy pybind11 scipy
```

### 2. Download and Build PanguLU

#### Clone PanguLU Source
```bash
# Navigate to the project directory
cd torch-pangulu

# Create third_party directory
mkdir -p third_party

# Clone PanguLU
git clone https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git third_party/PanguLU
```

#### Configure PanguLU Build
```bash
cd third_party/PanguLU

# Edit make.inc with your system paths
cp make.inc make.inc.backup

# Update paths in make.inc:
# - Set OPENBLAS_INC and OPENBLAS_LIB to your OpenBLAS installation
# - Set CUDA_INC and CUDA_LIB if using CUDA
# - Set METIS paths if using METIS
```

#### Example make.inc configuration:

**For Linux with OpenBLAS:**
```make
OPENBLAS_INC = -I/usr/include/openblas
OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu -lopenblas
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64
```

**For macOS with Homebrew:**
```make
OPENBLAS_INC = -I/opt/homebrew/opt/openblas/include
OPENBLAS_LIB = -L/opt/homebrew/opt/openblas/lib -lopenblas
MPICCFLAGS = $(OPENBLAS_INC) $(OPENBLAS_LIB) -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -lpthread -lm
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64
```

**For CUDA systems:**
```make
CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcudart -lcusparse
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN
```

#### Build PanguLU
```bash
# Build the library
make -j$(nproc)

# Verify build success
ls lib/  # Should contain libpangulu.a and libpangulu.so
```

### 3. Build Torch-PanguLU Extension

#### Using setuptools (Recommended):
```bash
# Return to project root
cd ../..

# Install in development mode
pip install -e .

# Or build extension in-place
python setup.py build_ext --inplace
```

#### Using CMake directly:
```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install
make install
```

### 4. Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Test basic functionality
python -c "import torch_pangulu; print(torch_pangulu._C.get_pangulu_info())"
```

## Build Configuration Options

### CMake Options

- `CMAKE_BUILD_TYPE`: `Release` (default), `Debug`, `RelWithDebInfo`
- `USE_CUDA`: `ON`/`OFF` - Enable CUDA support
- `USE_MPI`: `ON`/`OFF` - Enable MPI support  
- `PANGULU_ROOT_DIR`: Path to PanguLU installation
- `Python_EXECUTABLE`: Specific Python interpreter to use

Example:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DPANGULU_ROOT_DIR=/path/to/pangulu
```

### PanguLU Build Flags

Set in `third_party/PanguLU/make.inc`:

- `-DCALCULATE_TYPE_R64`: Double precision real numbers (recommended)
- `-DCALCULATE_TYPE_R32`: Single precision real numbers
- `-DCALCULATE_TYPE_CR64`: Double precision complex numbers
- `-DCALCULATE_TYPE_CR32`: Single precision complex numbers
- `-DGPU_OPEN`: Enable GPU support
- `-DMETIS`: Enable METIS reordering
- `-DPANGULU_MC64`: Enable MC64 scaling
- `-DPANGULU_LOG_INFO`: Verbose logging

## CUDA Support Build Instructions

### CUDA Prerequisites

**System Requirements for CUDA:**
- NVIDIA GPU with Compute Capability 6.0+ (GTX 10 series or newer)
- CUDA Toolkit 11.0+ (CUDA 12.x recommended)
- Compatible NVIDIA driver (470.x+ for CUDA 11, 520.x+ for CUDA 12)

**Check CUDA Installation:**
```bash
# Verify CUDA is available
nvcc --version
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### CUDA Build Options

There are two approaches for CUDA support:

#### Option 1: CPU-Only Build (Recommended - Stable)
**Status:** ✅ **Working** - Production ready with excellent numerical accuracy

```bash
# Use the automated script (recommended)
bash scripts/setup_pangulu.sh --cpu-only

# Or manually configure
cd third_party/PanguLU
# Edit make.inc to disable GPU support
sed -i 's/-DGPU_OPEN//' make.inc
make clean && make -j$(nproc)

# Build torch extension
cd ../..
python setup.py build_ext --inplace
```

**Features:**
- ✅ Stable and thoroughly tested
- ✅ Excellent numerical accuracy (1e-15 residuals)
- ✅ Full MPI support
- ✅ OpenMP parallelization

#### Option 2: CUDA-Enabled Build (Experimental)
**Status:** ⚠️ **Experimental** - PanguLU CUDA kernels have compatibility issues

```bash
# Use the automated script with CUDA
bash scripts/setup_pangulu.sh --enable-cuda

# Or manually configure (see detailed steps below)
```

**Current Issues:**
- PanguLU's CUDA kernels use `atomicAdd` with double precision, requiring Compute Capability 6.0+
- Some kernel implementations have compatibility issues with modern CUDA versions
- May require patching PanguLU's CUDA code for newer GPUs

### Automated CUDA Setup Script

We provide an automated script that handles all CUDA configuration:

```bash
# For CPU-only build (recommended)
bash scripts/setup_pangulu.sh --cpu-only

# For experimental CUDA build
bash scripts/setup_pangulu.sh --enable-cuda

# For CUDA build with specific CUDA path
bash scripts/setup_pangulu.sh --enable-cuda --cuda-path=/usr/local/cuda
```

### Manual CUDA Configuration

If you prefer manual configuration, follow these detailed steps:

#### Step 1: Configure PanguLU for CUDA

Edit `third_party/PanguLU/make.inc`:

```make
# CUDA Configuration
CUDA_PATH = /usr/local/cuda                    # Adjust path if needed
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse

# Compiler settings
NVCC = nvcc -O3
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg \
            -gencode=arch=compute_60,code=sm_60 \
            -gencode=arch=compute_70,code=sm_70 \
            -gencode=arch=compute_80,code=sm_80 \
            $(CUDA_INC)

# Enable GPU support (experimental)
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN
```

**Common CUDA Paths:**
- Ubuntu/Debian: `/usr/lib/cuda` or `/usr/local/cuda`
- CentOS/RHEL: `/usr/local/cuda`
- Custom install: Check `which nvcc` and use parent directory

#### Step 2: Update Torch Extension for CUDA

The `setup.py` automatically detects CUDA availability. To force enable/disable:

```bash
# Force enable CUDA (experimental)
export TORCH_CUDA_FORCE=1
python setup.py build_ext --inplace

# Force disable CUDA (stable)
export TORCH_CUDA_FORCE=0
python setup.py build_ext --inplace
```

#### Step 3: Build and Test

```bash
# Clean and rebuild PanguLU
cd third_party/PanguLU
make clean && make -j$(nproc)

# Build torch extension
cd ../..
python setup.py build_ext --inplace

# Test CUDA detection
python -c "
import torch_pangulu
info = torch_pangulu._C.get_pangulu_info()
print('CUDA support:', info['cuda_support'])
print('Available:', info['available'])
"
```

### CUDA Troubleshooting

#### Common CUDA Issues:

**1. CUDA Not Found During Build:**
```bash
# Check CUDA installation
ls /usr/local/cuda/bin/nvcc  # or /usr/lib/cuda/bin/nvcc

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

**2. Compilation Errors with atomicAdd:**
```
error: no instance of overloaded function "atomicAdd" matches the argument list
        argument types are: (double *, double)
```

**Solution:** This is a known issue with PanguLU's CUDA kernels. Use CPU-only build:
```bash
bash scripts/setup_pangulu.sh --cpu-only
```

**3. Compute Capability Issues:**
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# For older GPUs (Compute < 6.0), use CPU-only build
```

**4. CUDA Runtime Version Mismatch:**
```bash
# Check CUDA runtime vs driver version
nvcc --version                    # Toolkit version
nvidia-smi | grep "CUDA Version"  # Driver version

# Update NVIDIA drivers if needed
sudo apt update && sudo apt install nvidia-driver-535  # or latest
```

### Performance Considerations

**CPU vs GPU Performance:**
- **CPU Version:** Optimized for large sparse matrices, excellent for most use cases
- **CUDA Version:** Potentially faster for very large matrices (>100,000 x 100,000) when stable
- **Memory:** CUDA requires substantial GPU memory for large problems

**Recommendations:**
1. **Start with CPU-only build** - it's stable and performs excellently
2. **Test CUDA only if** you have large matrices and modern hardware
3. **Use CPU build for production** until CUDA stability improves

### CUDA Validation Tests

After building with CUDA support, run these validation tests:

```bash
# Basic functionality test
python -c "
import torch
import torch_pangulu

# Test info
info = torch_pangulu._C.get_pangulu_info()
print('CUDA support detected:', info['cuda_support'])

# Create test matrix
n = 100
indices = torch.randint(0, n, (2, 1000))
values = torch.randn(1000, dtype=torch.float64)
A = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64).coalesce()
b = torch.randn(n, dtype=torch.float64)

# Test solve
try:
    x = torch_pangulu.sparse_lu_solve(A, b)
    residual = torch.norm(torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b)
    print(f'Residual: {residual:.2e}')
    if residual < 1e-10:
        print('✅ Test passed!')
    else:
        print('⚠️  High residual - check configuration')
except Exception as e:
    print(f'❌ Test failed: {e}')
"
```

## Troubleshooting

### Common Issues

#### 1. MPI Not Found
```bash
# Check MPI installation
which mpicc
mpicc --version

# Add to PATH if needed
export PATH=$PATH:/usr/lib64/openmpi/bin
```

#### 2. CUDA Not Found (if using CUDA)
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

#### 3. OpenBLAS Not Found
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# CentOS/RHEL
sudo yum install openblas-devel

# macOS
brew install openblas
```

#### 4. Compilation Errors

**Undefined symbols for PanguLU functions:**
- Ensure PanguLU was built successfully: `ls third_party/PanguLU/lib/`
- Check that `libpangulu.a` or `libpangulu.so` exists
- Verify architecture compatibility (x86_64 vs ARM64)

**PyTorch header not found:**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__file__)"
python -c "import torch; print(torch.utils.cmake_prefix_path)"

# Reinstall if needed
pip install --force-reinstall torch
```

#### 5. Runtime Issues

**MPI initialization failed:**
- Check MPI environment: `mpirun --version`
- For single-node usage, try: `export OMPI_MCA_btl_vader_single_copy_mechanism=none`

**CUDA out of memory:**
- Reduce matrix size or use CPU-only version
- Check GPU memory: `nvidia-smi`

### Performance Optimization

1. **Use appropriate BLAS library:**
   - Intel MKL (fastest on Intel CPUs)
   - OpenBLAS (good general performance)
   - Accelerate Framework (macOS)

2. **Tune MPI settings:**
   - Set appropriate number of MPI processes
   - Use CPU binding: `mpirun --bind-to core`

3. **CUDA optimization:**
   - Use recent CUDA version (11.0+)
   - Ensure GPU has sufficient memory
   - Consider mixed precision for larger problems

## Testing

### Run Full Test Suite
```bash
python -m pytest tests/ -v --tb=short
```

### Run Specific Tests
```bash
# Test import only
python -c "import torch_pangulu"

# Test basic functionality
python tests/test_sparse_lu.py

# Benchmark performance
python examples/benchmark.py
```

### Debugging

Enable verbose logging:
```bash
export PANGULU_LOG_LEVEL=INFO
python your_script.py
```

Build debug version:
```bash
python setup.py build_ext --inplace --debug
```

## Performance Considerations

- **Matrix size**: PanguLU is optimized for large sparse matrices (>10,000 x 10,000)
- **Sparsity**: Best performance with sparsity 0.1% - 10%
- **Block size**: Tune the `nb` parameter in PanguLU initialization
- **Memory**: Ensure sufficient RAM (problem size dependent)
- **CPU cores**: Use multiple MPI processes for large problems
- **GPU memory**: CUDA version requires substantial GPU memory

For specific performance tuning advice, refer to the PanguLU User's Guide included in the repository.