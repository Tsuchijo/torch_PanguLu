# Torch-PanguLU

A PyTorch C++ extension that integrates PanguLU fast sparse LU decomposition library with PyTorch sparse tensors.

## Overview

This project provides a seamless interface between PyTorch's sparse tensor functionality and PanguLU's high-performance sparse LU decomposition capabilities. **PanguLU integration is now complete** with excellent numerical accuracy (residuals ~1e-15) and comprehensive MPI support for distributed computing.

## Features

- **✅ Complete PanguLU Integration**: Full sparse LU decomposition and solving
- **✅ PyTorch Integration**: Native support for PyTorch sparse tensors with automatic CSR conversion
- **✅ Excellent Accuracy**: Machine precision results (residual norms 1e-15 to 1e-13)
- **✅ High Performance**: Leverages PanguLU's optimized sparse LU implementation
- **✅ MPI Support**: Distributed computing for large-scale problems
- **✅ Real-time Logging**: Detailed PanguLU performance timing information
- **✅ Comprehensive Testing**: Extensive validation against reference solvers
- **⚠️ GPU Acceleration**: Framework ready, requires CUDA-enabled build
- **✅ Multiple Precision**: Support for double precision (single precision ready)
- **✅ Easy Installation**: Standard Python package installation with automated setup

## Prerequisites

### Required Dependencies

- Python 3.7+
- PyTorch >= 1.9.0
- MPI library (OpenMPI 4.1.2+ recommended)
- C++17 compatible compiler (GCC 7.0+ or Clang 10.0+)
- CMake 3.12+
- OpenBLAS or Intel MKL
- Pybind11 >= 2.6.0

### Optional Dependencies

- CUDA toolkit (for GPU support)
- OpenMP (for additional parallelization)
- METIS library (for matrix reordering)

## Installation

### Quick Start (Automated Setup)

We provide an automated setup script that handles all configuration and building:

```bash
# CPU-only build (recommended - stable and fast)
bash scripts/setup_pangulu.sh --cpu-only

# Experimental CUDA build (may have compatibility issues)
bash scripts/setup_pangulu.sh --enable-cuda

# View all options
bash scripts/setup_pangulu.sh --help
```

The automated script will:
- ✅ Check system requirements and dependencies
- ✅ Create virtual environment and install Python packages
- ✅ Clone and configure PanguLU with optimal settings
- ✅ Build both PanguLU and torch-pangulu extension
- ✅ Run validation tests to ensure everything works

**Build Status:**
- **CPU-Only**: ✅ **Stable** - Production ready with excellent numerical accuracy
- **CUDA**: ⚠️ **Experimental** - PanguLU CUDA kernels have compatibility issues with modern CUDA

### Manual Setup Guide

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential cmake git openmpi-bin openmpi-common libopenmpi-dev libopenblas-dev
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake git openmpi openmpi-devel openblas-devel
```

**macOS:**
```bash
brew install open-mpi openblas libomp cmake git
```

#### 2. Setup Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch numpy scipy pybind11 pytest
```

#### 3. Clone and Build PanguLU

```bash
# Clone PanguLU to third_party directory
mkdir -p third_party
git clone https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git third_party/PanguLU

# Configure PanguLU build
cd third_party/PanguLU
cp make.inc make.inc.backup

# Update make.inc with system paths (example for Ubuntu/Debian):
cat > make.inc << 'EOF'
COMPILE_LEVEL = -O3
CC = gcc $(COMPILE_LEVEL)
MPICC = mpicc $(COMPILE_LEVEL)
OPENBLAS_INC = -I/usr/include/x86_64-linux-gnu/openblas-pthread/
OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas
MPICCFLAGS = $(OPENBLAS_INC) $(OPENBLAS_LIB) -fopenmp -lpthread -lm
MPICCLINK = $(OPENBLAS_LIB)
METISFLAGS = 
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64
EOF

# Build PanguLU
make -j$(nproc)
cd ../..
```

#### 4. Build Torch-PanguLU Extension

```bash
# Build the C++ extension
python setup.py build_ext --inplace

# Test the installation
python -c "import torch_pangulu; print('Success!')"
python -c "import torch_pangulu; print(torch_pangulu._C.get_pangulu_info())"
```

### Quick Setup (Alternative)

If you have the dependencies installed, you can use a one-liner setup:

```bash
# Complete setup in one command
git clone https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git third_party/PanguLU && \
cd third_party/PanguLU && \
echo -e "COMPILE_LEVEL = -O3\nCC = gcc \$(COMPILE_LEVEL)\nMPICC = mpicc \$(COMPILE_LEVEL)\nOPENBLAS_INC = -I/usr/include/x86_64-linux-gnu/openblas-pthread/\nOPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas\nMPICCFLAGS = \$(OPENBLAS_INC) \$(OPENBLAS_LIB) -fopenmp -lpthread -lm\nMPICCLINK = \$(OPENBLAS_LIB)\nMETISFLAGS = \nPANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64" > make.inc && \
make -j$(nproc) && \
cd ../.. && \
python setup.py build_ext --inplace
```

For detailed build instructions and troubleshooting, see **[BUILD.md](BUILD.md)**.

## CUDA Support

### Current CUDA Status

- **CPU Version**: ✅ **Stable and Recommended** - Excellent performance and numerical accuracy
- **CUDA Version**: ⚠️ **Experimental** - Known compatibility issues with PanguLU CUDA kernels

### CUDA Build Options

#### Option 1: Use Automated Script (Recommended)
```bash
# For experimental CUDA build
bash scripts/setup_pangulu.sh --enable-cuda

# For stable CPU-only build (recommended)
bash scripts/setup_pangulu.sh --cpu-only
```

#### Option 2: Manual CUDA Configuration

**Prerequisites:**
- NVIDIA GPU with Compute Capability 6.0+ (GTX 10 series or newer)
- CUDA Toolkit 11.0+ installed
- Compatible NVIDIA drivers

**Enable CUDA in PanguLU:**
```bash
# Edit third_party/PanguLU/make.inc
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN
CUDA_PATH = /usr/local/cuda  # or /usr/lib/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse
```

**Force CUDA in torch extension:**
```bash
export TORCH_CUDA_FORCE=1
python setup.py build_ext --inplace
```

### Known CUDA Issues

1. **Compilation Errors**: PanguLU CUDA kernels may fail to compile with modern CUDA versions due to `atomicAdd` double precision requirements
2. **Kernel Compatibility**: Some CUDA kernels have compatibility issues with newer GPU architectures
3. **Memory Requirements**: CUDA version requires substantial GPU memory

### CUDA Troubleshooting

If you encounter CUDA build issues:

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Fall back to stable CPU build
bash scripts/setup_pangulu.sh --cpu-only --force-rebuild
```

**Recommendation**: Use the stable CPU-only build unless you specifically need CUDA and have compatible hardware.

## Development Status

✅ **PanguLU Integration Complete!** 

**Current Status:**
- ✅ Project structure and build system complete
- ✅ Python/C++ integration framework ready
- ✅ Full PanguLU API integration implemented
- ✅ **Real sparse LU solving with excellent accuracy** (residuals ~1e-15)
- ✅ MPI support and parallel processing
- ✅ Comprehensive testing and validation

**What Works:**
- Full PanguLU sparse LU decomposition and solving
- PyTorch sparse tensor integration with CSR conversion
- **Automatic device detection** from input tensors (CPU/GPU)
- **Device override functionality** for explicit control
- MPI-based parallel processing
- Machine precision numerical accuracy
- Comprehensive error handling and validation
- Real-time PanguLU performance logging
- Test framework with extensive coverage

**Performance Results:**
- **Numerical Accuracy**: Residual norms of 1e-15 to 1e-13
- **Speed**: Efficient factorization and solve phases
- **Scalability**: Handles matrices from 10x10 to 500x500+ elements
- **Compatibility**: Matches SciPy reference solver accuracy

**What's Optional:**
- GPU/CUDA support (framework ready, requires CUDA-enabled build)
- Factor extraction interface (PanguLU doesn't expose L/U factors directly)
- Advanced optimization features

For a detailed development roadmap and contribution guidelines, see **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Quick Start

```python
import torch
import torch_pangulu
import numpy as np
from scipy.sparse import random as sparse_random

# Create a sparse matrix
n = 1000
scipy_matrix = sparse_random(n, n, density=0.01, format='coo', dtype=np.float64)

# Make it symmetric positive definite for stability
scipy_matrix = scipy_matrix + scipy_matrix.T
scipy_matrix.setdiag(scipy_matrix.diagonal() + n * 0.1)
scipy_matrix = scipy_matrix.tocoo()  # Ensure COO format

# Convert to PyTorch sparse tensor
indices = torch.from_numpy(np.vstack([scipy_matrix.row, scipy_matrix.col])).long()
values = torch.from_numpy(scipy_matrix.data).double()
A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

# Create right-hand side
b = torch.randn(n, dtype=torch.float64)

# Solve the linear system using PanguLU
print("Solving with PanguLU...")
x = torch_pangulu.sparse_lu_solve(A, b)

# Verify solution accuracy
residual = torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b
residual_norm = torch.norm(residual)
print(f"Residual norm: {residual_norm:.2e}")  # Expect ~1e-15 accuracy!

# Check PanguLU status
info = torch_pangulu._C.get_pangulu_info()
print(f"PanguLU available: {info['available']}")
print(f"MPI support: {info['mpi_support']}")
```

**Expected Output:**
```
Solving with PanguLU...
[PanguLU Info] n=1000 nnz=XX nb=64 mpi_process=1 preprocessing_thread=1
[PanguLU Info] ... (detailed timing information)
Residual norm: 1.23e-15
PanguLU available: True
MPI support: True
```

### Device Detection Examples

The library automatically detects tensor devices and can use GPU acceleration when available:

```python
import torch
import torch_pangulu

# Create test matrix
n = 100
indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
A = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64).coalesce()
b = torch.randn(n, dtype=torch.float64)

# 1. Auto-detection with CPU tensors
x1 = torch_pangulu.sparse_lu_solve(A, b)  # Uses CPU automatically
print(f"Solution device: {x1.device}")  # cpu

# 2. Explicit device override
x2 = torch_pangulu.sparse_lu_solve(A, b, device='cpu')  # Force CPU
x3 = torch_pangulu.sparse_lu_solve(A, b, device=torch.device('cpu'))  # torch.device object

# 3. GPU tensor handling (if CUDA available)
if torch.cuda.is_available():
    A_gpu = A.cuda()
    b_gpu = b.cuda()
    
    # Auto-detects GPU input, uses GPU if PanguLU supports it, else falls back to CPU
    x_gpu = torch_pangulu.sparse_lu_solve(A_gpu, b_gpu)
    
    # Force CPU computation even with GPU inputs
    x_cpu = torch_pangulu.sparse_lu_solve(A_gpu, b_gpu, device='cpu')

# 4. Check device capabilities
info = torch_pangulu._C.get_pangulu_info()
print(f"CUDA support compiled: {info['cuda_support']}")
print(f"CUDA runtime available: {info['cuda_available']}")
print(f"Auto device detection: {info['auto_device_detection']}")
```

## API Reference

### Core Functions

#### `sparse_lu_solve(sparse_matrix, rhs, factorize=True, device=None)`

Solve a sparse linear system Ax = b using PanguLU.

**Parameters:**
- `sparse_matrix` (torch.Tensor): Sparse coefficient matrix in COO format
- `rhs` (torch.Tensor): Right-hand side vector or matrix
- `factorize` (bool): Whether to perform factorization or reuse cached factors
- `device` (torch.device, str, optional): Device to perform computation on. If None, automatically detects from input tensors.

**Returns:**
- `torch.Tensor`: Solution vector/matrix x on the same device as inputs

#### `sparse_lu_factorize(sparse_matrix, device=None)`

Perform LU factorization and return factors.

**Parameters:**
- `sparse_matrix` (torch.Tensor): Sparse matrix to factorize
- `device` (torch.device, str, optional): Device to perform computation on. If None, automatically detects from input tensor.

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: L and U factors (if available)

### Utility Functions

#### `get_pangulu_info()`

Get information about PanguLU configuration and capabilities.

#### `clear_factorization()`

Clear any cached factorization data.

#### `has_cached_factorization()`

Check if factorization is cached for reuse.

## Performance Considerations

### Numerical Accuracy
- **Excellent precision**: Residual norms typically 1e-15 to 1e-13
- **Reference quality**: Results match SciPy sparse solvers to machine precision
- **Robust**: Handles various matrix conditions and sparsity patterns

### Matrix Format
- Input matrices should be in COO (coordinate) sparse format
- Matrices are automatically converted to CSR format internally
- For best performance, ensure matrices are coalesced
- Symmetric positive definite matrices work best

### Memory Usage
- PanguLU may require significant memory for large problems
- Memory usage scales with matrix size and fill-in during factorization
- Consider using distributed computing for very large matrices
- Clear cached factorizations when memory is constrained

### Performance Characteristics
- **Small matrices** (10-100): Fast initialization, minimal overhead
- **Medium matrices** (100-1000): Good balance of speed and accuracy
- **Large matrices** (1000+): Leverages PanguLU's optimized algorithms
- **Sparse matrices**: Performance scales with sparsity (better for sparser matrices)

### Parallelization
- MPI parallelization is handled automatically
- Set `OMP_NUM_THREADS` for OpenMP parallelization  
- Single-threaded performance is already excellent
- GPU acceleration requires CUDA-enabled build

### Timing Information
PanguLU provides detailed timing breakdowns:
- **Preprocessing**: Matrix reordering and symbolic factorization
- **Numeric factorization**: LU decomposition computation
- **Solving**: Forward/backward substitution
- **Total time**: Complete solution process

Example timing for 500x500 matrix with ~10K nonzeros:
- Preprocessing: ~0.11s
- Numeric factorization: ~0.01s  
- Solving: ~0.0004s

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Complete linear system solving with accuracy verification

### Running the Examples

```bash
# Activate your environment
source venv/bin/activate

# Run the basic usage example
PYTHONPATH=/path/to/torch_PanguLu python examples/basic_usage.py
```

**Example Output:**
```
Torch-PanguLU Basic Usage Example
========================================
PanguLU Info: {'available': True, 'version': '4.2.0', 'cuda_support': False, 'mpi_support': True, 'openmp_support': True, 'mpi_initialized': False}

1. Creating test sparse matrix...
   Matrix size: torch.Size([500, 500])
   Non-zeros: 10406
   Density: 4.16%

2. Creating right-hand side vector...
   RHS norm: 1108.602803

3. Solving using PanguLU...
   Solution computed successfully!
   Residual norm: 5.06e-13
   Solution error: 1.01e-14

4. Reference solution using SciPy...
   SciPy solution error: 2.02e-14
   PanguLU vs SciPy difference: 2.12e-14

[PanguLU Info] n=500 nnz=10406 nb=64 mpi_process=1 preprocessing_thread=1
[PanguLU Info] 1.2 PanguLU MC64 reordering time is 0.000170 s.
[PanguLU Info] 3.2 PanguLU METIS reordering time is 0.000231 s.
[PanguLU Info] Reordering time is 0.000547 s.
[PanguLU Info] 4 PanguLU A+AT (before symbolic) time is 0.000074 s.
[PanguLU Info] Symbolic nonzero count is 208218.
[PanguLU Info] 5 PanguLU symbolic time is 0.002013 s.
[PanguLU Info] Symbolic factorization time is 0.002092 s.
[PanguLU Info] 6 PanguLU transpose reordered matrix time is 0.000036 s.
[PanguLU Info] 7 PanguLU generate full symbolic matrix time is 0.001236 s.
[PanguLU Info] Preprocessing time is 0.110728 s.
[PanguLU Info] Numeric factorization time is 0.011321 s.
[PanguLU Info] Solving time is 0.000421 s.

Example completed!
```

## Testing

Run the test suite:

```bash
# Install pytest if not already installed
pip install pytest

# Run the full test suite
python -m pytest tests/ -v

# Run specific tests directly
python tests/test_sparse_lu.py

# Test just the solve functionality (recommended)
python -c "
import torch_pangulu
import torch
import numpy as np
from scipy.sparse import random as sparse_random

print('Testing PanguLU solve functionality...')
n = 20
scipy_matrix = sparse_random(n, n, density=0.1, format='coo', dtype=np.float64)
scipy_matrix = scipy_matrix + scipy_matrix.T
scipy_matrix.setdiag(scipy_matrix.diagonal() + n * 0.1)
scipy_matrix = scipy_matrix.tocoo()

indices = torch.from_numpy(np.vstack([scipy_matrix.row, scipy_matrix.col])).long()
values = torch.from_numpy(scipy_matrix.data).double()
A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

x_true = torch.randn(n, dtype=torch.float64)
b = torch.sparse.mm(A, x_true.unsqueeze(1)).squeeze()

x_solved = torch_pangulu.sparse_lu_solve(A, b)
residual_norm = torch.norm(torch.sparse.mm(A, x_solved.unsqueeze(1)).squeeze() - b)

print(f'Residual norm: {residual_norm:.2e}')
print('✅ Test passed!' if residual_norm < 1e-10 else '❌ Test failed')
"
```

**Expected Test Results:**
- **Numerical accuracy tests**: Residual norms of 1e-15 to 1e-13
- **Import tests**: All modules load successfully  
- **Integration tests**: PanguLU functions execute without errors
- **Validation tests**: Results match reference solvers

Note: Some tests may be skipped if they test advanced features not yet implemented (like factor extraction).

## Configuration

### Build Configuration

The build system supports several configuration options:

```bash
# Enable CUDA support
export CMAKE_ARGS="-DUSE_CUDA=ON"

# Specify PanguLU paths
export PANGULU_INCLUDE_DIR=/custom/path/include
export PANGULU_LIB_DIR=/custom/path/lib

# Set precision type
export CMAKE_ARGS="-DCALCULATE_TYPE_R64=ON"  # Double precision real
```

### Runtime Configuration

Environment variables for runtime behavior:

```bash
# MPI configuration
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# OpenMP threads
export OMP_NUM_THREADS=8

# PanguLU logging
export PANGULU_LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

1. **Import Error**: `ImportError: libpangulu.so: cannot open shared object file`
   ```bash
   # Ensure PanguLU was built correctly
   ls third_party/PanguLU/lib/libpangulu.so
   
   # Rebuild the extension with correct paths
   python setup.py build_ext --inplace
   ```

2. **MPI Compilation Errors**: `fatal error: mpi.h: No such file or directory`
   ```bash
   # Install MPI development headers
   sudo apt install openmpi-bin openmpi-common libopenmpi-dev
   
   # Verify MPI installation
   mpicc --version
   which mpicc
   ```

3. **PyTorch Version Issues**: C++17 compiler errors
   ```bash
   # Ensure you have a modern compiler
   gcc --version  # Should be 7.0+
   
   # Reinstall PyTorch if needed
   pip install --force-reinstall torch
   ```

4. **OpenBLAS Linking Errors**: Library not found errors
   ```bash
   # Install OpenBLAS development libraries
   sudo apt install libopenblas-dev
   
   # Check installation
   pkg-config --libs openblas
   ```

5. **PanguLU Build Errors**: Compilation fails in third_party/PanguLU
   ```bash
   # Check make.inc configuration
   cd third_party/PanguLU
   cat make.inc
   
   # Ensure paths are correct for your system
   # See examples in the installation section
   ```

### Runtime Issues

6. **MPI Initialization Warnings**: Multiple MPI init calls
   ```bash
   # This is normal - MPI is initialized automatically
   # No action needed, warnings can be ignored
   ```

7. **Memory Issues**: Large matrix problems
   ```bash
   # Reduce matrix size or increase available memory
   # PanguLU requires significant memory for factorization
   ```

### Debug Mode

Enable debug output and check configuration:

```python
import torch_pangulu

# Check PanguLU integration status
info = torch_pangulu._C.get_pangulu_info()
print(f"PanguLU configuration: {info}")

# Expected output:
# {'available': True, 'version': '4.2.0', 'cuda_support': False, 
#  'mpi_support': True, 'openmp_support': True, 'mpi_initialized': False}

# Test with a small matrix
import torch
import numpy as np
from scipy.sparse import random as sparse_random

n = 10
scipy_matrix = sparse_random(n, n, density=0.3, format='coo', dtype=np.float64)
scipy_matrix = scipy_matrix + scipy_matrix.T
scipy_matrix.setdiag(scipy_matrix.diagonal() + n * 0.1)
scipy_matrix = scipy_matrix.tocoo()

indices = torch.from_numpy(np.vstack([scipy_matrix.row, scipy_matrix.col])).long()
values = torch.from_numpy(scipy_matrix.data).double()
A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
b = torch.randn(n, dtype=torch.float64)

try:
    x = torch_pangulu.sparse_lu_solve(A, b)
    print("✅ PanguLU working correctly!")
    print(f"Solution norm: {torch.norm(x):.2e}")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Performance Verification

Compare with SciPy to verify correctness:

```python
import torch_pangulu
import scipy.sparse.linalg as spla

# ... create A, b as above ...

# Solve with PanguLU
x_pangulu = torch_pangulu.sparse_lu_solve(A, b)

# Solve with SciPy reference
A_scipy = scipy_matrix  # Use the original scipy matrix
b_numpy = b.numpy()
x_scipy = spla.spsolve(A_scipy, b_numpy)

# Compare solutions
diff = torch.norm(x_pangulu - torch.from_numpy(x_scipy))
print(f"PanguLU vs SciPy difference: {diff:.2e}")  # Should be ~1e-14
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

PanguLU is distributed under its own license terms. Please refer to the PanguLU repository for licensing information.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{torch_pangulu,
  title={Torch-PanguLU: PyTorch Integration for PanguLU Sparse Solvers},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/torch-pangulu}
}
```

Also consider citing the original PanguLU paper and PyTorch.

## Acknowledgments

- PanguLU development team at SuperScientific Software Laboratory
- PyTorch development team
- Contributors to this project