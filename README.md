# Torch-PanguLU

A PyTorch C++ extension that integrates PanguLU fast sparse LU decomposition library with PyTorch sparse tensors.

## Overview

This project provides a seamless interface between PyTorch's sparse tensor functionality and PanguLU's high-performance sparse LU decomposition capabilities. PanguLU is designed for distributed computing environments and offers significant speedups for large sparse linear systems.

## Features

- **PyTorch Integration**: Native support for PyTorch sparse tensors
- **High Performance**: Leverages PanguLU's optimized sparse LU implementation
- **Distributed Computing**: MPI support for large-scale problems
- **GPU Acceleration**: Optional CUDA support (when available)
- **Multiple Precision**: Support for single and double precision
- **Easy Installation**: Standard Python package installation

## Prerequisites

### Required Dependencies

- Python 3.7+
- PyTorch >= 1.9.0
- MPI library (OpenMPI 4.1.2+ recommended)
- C++14 compatible compiler
- CMake 3.12+

### Optional Dependencies

- CUDA toolkit (for GPU support)
- OpenMP (for additional parallelization)
- METIS library (for matrix reordering)

## Installation

### Quick Setup (Recommended)

Use the automated setup script:

```bash
# Run the setup script to automatically configure and build PanguLU
./scripts/setup_pangulu.sh

# Install the PyTorch extension
pip install -e .
```

### Manual Installation

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

#### 2. Setup PanguLU

```bash
# Clone PanguLU (automatically placed in third_party/ and excluded from git)
mkdir -p third_party
git clone https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git third_party/PanguLU

# Configure and build
cd third_party/PanguLU
# Edit make.inc with your system paths (see BUILD.md for details)
make -j$(nproc)
cd ../..
```

#### 3. Install Torch-PanguLU

```bash
# Install Python dependencies
pip install torch numpy pybind11 scipy pytest

# Install in development mode
pip install -e .
```

For detailed build instructions and troubleshooting, see **[BUILD.md](BUILD.md)**.

## Development Status

⚠️ **This project is currently in active development.** 

**Current Status:**
- ✅ Project structure and build system complete
- ✅ Python/C++ integration framework ready
- ✅ Basic API design implemented
- ⚠️ **PanguLU integration partially complete** (mock implementation)
- ❌ Full functionality not yet available

**What Works:**
- Project builds successfully 
- Python module imports correctly
- Basic tensor validation and conversion
- Test framework and examples

**What's Missing:**
- Complete PanguLU function integration
- Real sparse LU solving (currently uses mock implementation)
- GPU/CUDA support
- Performance optimization

For a detailed development roadmap and contribution guidelines, see **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Quick Start

```python
import torch
import torch_pangulu
import numpy as np
from scipy.sparse import random

# Create a sparse matrix
n = 1000
scipy_matrix = random(n, n, density=0.01, format='coo')
indices = torch.from_numpy(np.vstack([scipy_matrix.row, scipy_matrix.col])).long()
values = torch.from_numpy(scipy_matrix.data).double()
A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

# Create right-hand side
b = torch.randn(n, dtype=torch.float64)

# Solve the linear system
x = torch_pangulu.sparse_lu_solve(A, b)

# Verify solution
residual = torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b
print(f"Residual norm: {torch.norm(residual):.2e}")
```

## API Reference

### Core Functions

#### `sparse_lu_solve(sparse_matrix, rhs, factorize=True)`

Solve a sparse linear system Ax = b using PanguLU.

**Parameters:**
- `sparse_matrix` (torch.Tensor): Sparse coefficient matrix in COO format
- `rhs` (torch.Tensor): Right-hand side vector or matrix
- `factorize` (bool): Whether to perform factorization or reuse cached factors

**Returns:**
- `torch.Tensor`: Solution vector/matrix x

#### `sparse_lu_factorize(sparse_matrix)`

Perform LU factorization and return factors.

**Parameters:**
- `sparse_matrix` (torch.Tensor): Sparse matrix to factorize

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

### Matrix Format
- Input matrices should be in COO (coordinate) sparse format
- Matrices are automatically converted to CSR format internally
- For best performance, ensure matrices are coalesced

### Memory Usage
- PanguLU may require significant memory for large problems
- Consider using distributed computing for very large matrices
- Clear cached factorizations when memory is constrained

### Parallelization
- MPI parallelization is handled automatically
- Set `OMP_NUM_THREADS` for OpenMP parallelization
- GPU acceleration requires CUDA-enabled build

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Simple linear system solving
- `performance_comparison.py`: Comparison with other solvers
- `distributed_example.py`: MPI distributed computing example

## Testing

Run the test suite:

```bash
python -m pytest tests/
# or
python tests/test_sparse_lu.py
```

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

1. **Import Error**: Ensure the C++ extension is compiled
   ```bash
   python setup.py build_ext --inplace
   ```

2. **MPI Errors**: Check MPI installation and configuration
   ```bash
   mpirun --version
   ```

3. **CUDA Issues**: Verify CUDA installation and compatibility
   ```bash
   nvcc --version
   nvidia-smi
   ```

4. **Memory Errors**: Reduce problem size or use distributed computing

### Debug Mode

Enable debug output:

```python
import torch_pangulu
info = torch_pangulu._C.get_pangulu_info()
print(f"PanguLU configuration: {info}")
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