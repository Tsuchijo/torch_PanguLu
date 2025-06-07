import torch
from typing import Tuple, Optional

try:
    from . import _C
except ImportError:
    raise ImportError(
        "torch_pangulu C++ extension is not compiled. "
        "Please run 'python setup.py build_ext --inplace' or 'pip install -e .'"
    )


def sparse_lu_factorize(sparse_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform LU factorization of a sparse matrix using PanguLU.
    
    Args:
        sparse_matrix: PyTorch sparse tensor in COO format
        
    Returns:
        Tuple of (L, U) factors as sparse tensors
        
    Raises:
        RuntimeError: If factorization fails
        ValueError: If input is not a sparse tensor
    """
    if not sparse_matrix.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    if sparse_matrix.layout != torch.sparse_coo:
        sparse_matrix = sparse_matrix.coalesce().to_sparse_coo()
    
    return _C.sparse_lu_factorize(sparse_matrix)


def sparse_lu_solve(
    sparse_matrix: torch.Tensor, 
    rhs: torch.Tensor,
    factorize: bool = True
) -> torch.Tensor:
    """
    Solve a sparse linear system using PanguLU.
    
    Args:
        sparse_matrix: PyTorch sparse tensor in COO format (coefficient matrix)
        rhs: Right-hand side tensor
        factorize: Whether to perform factorization (True) or use pre-computed factors
        
    Returns:
        Solution tensor x such that sparse_matrix @ x = rhs
        
    Raises:
        RuntimeError: If solve fails
        ValueError: If inputs are incompatible
    """
    if not sparse_matrix.is_sparse:
        raise ValueError("Matrix must be a sparse tensor")
    
    if sparse_matrix.layout != torch.sparse_coo:
        sparse_matrix = sparse_matrix.coalesce().to_sparse_coo()
    
    # Check dimensions
    if sparse_matrix.size(0) != rhs.size(0):
        raise ValueError(
            f"Matrix rows ({sparse_matrix.size(0)}) must match RHS rows ({rhs.size(0)})"
        )
    
    return _C.sparse_lu_solve(sparse_matrix, rhs, factorize)