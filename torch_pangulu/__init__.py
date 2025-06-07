"""
Torch-PanguLU: PyTorch C++ extension for PanguLU sparse LU decomposition

This package provides efficient sparse LU decomposition for PyTorch sparse tensors
using the PanguLU library for high-performance computing.

Features:
- High-performance sparse LU decomposition
- PyTorch sparse tensor integration
- GPU acceleration support (CUDA)
- Distributed computing with MPI
- Multiple precision support

Example:
    import torch
    import torch_pangulu
    
    # Create sparse matrix
    A = torch.sparse_coo_tensor([[0, 1], [1, 0]], [2.0, 1.0], (2, 2))
    b = torch.tensor([1.0, 2.0])
    
    # Solve linear system
    x = torch_pangulu.sparse_lu_solve(A, b)
"""

from .sparse_lu import sparse_lu_solve, sparse_lu_factorize

# Version information
__version__ = "0.1.0-dev"
__author__ = "Torch-PanguLU Contributors"
__license__ = "MIT"

# Public API
__all__ = [
    "sparse_lu_solve",
    "sparse_lu_factorize",
    # Utility functions (available when C++ extension is built)
    # "get_pangulu_info",
    # "clear_factorization", 
    # "has_cached_factorization",
]

# Try to import C++ extension
try:
    from . import _C
    
    # Add C++ functions to public API when available
    __all__.extend([
        "get_pangulu_info",
        "clear_factorization",
        "has_cached_factorization",
    ])
    
    # Make C++ functions available at package level
    get_pangulu_info = _C.get_pangulu_info
    clear_factorization = _C.clear_factorization
    has_cached_factorization = _C.has_cached_factorization
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"torch_pangulu C++ extension not available: {e}\n"
        "Run 'python setup.py build_ext --inplace' to build the extension.",
        ImportWarning
    )
    
    # Provide stub functions for development
    def get_pangulu_info():
        return {"available": False, "error": "C++ extension not built"}
    
    def clear_factorization():
        pass
    
    def has_cached_factorization():
        return False