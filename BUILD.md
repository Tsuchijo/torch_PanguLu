# Build Instructions

This document provides comprehensive instructions for building the Torch-PanguLU project on different platforms.

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

1. **CUDA Toolkit**: Version 11.0+ (for GPU acceleration)
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