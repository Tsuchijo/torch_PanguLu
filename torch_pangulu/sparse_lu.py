import torch
from typing import Tuple, Optional, Union

try:
    from . import _C
except ImportError:
    raise ImportError(
        "torch_pangulu C++ extension is not compiled. "
        "Please run 'python setup.py build_ext --inplace' or 'pip install -e .'"
    )


def sparse_lu_factorize(
    sparse_matrix: torch.Tensor, 
    device: Optional[Union[torch.device, str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform LU factorization of a sparse matrix using PanguLU.
    
    Args:
        sparse_matrix: PyTorch sparse tensor in COO format
        device: Device to perform computation on. If None, automatically detects from input tensor.
                For GPU tensors, uses GPU if CUDA support is available, otherwise falls back to CPU.
        
    Returns:
        Tuple of (L, U) factors as sparse tensors on the same device as the input
        
    Raises:
        RuntimeError: If factorization fails
        ValueError: If input is not a sparse tensor
    """
    if not sparse_matrix.is_sparse:
        raise ValueError("Input must be a sparse tensor")
    
    if sparse_matrix.layout != torch.sparse_coo:
        sparse_matrix = sparse_matrix.coalesce().to_sparse_coo()
    
    # Convert device to torch.device if string
    torch_device = None
    if device is not None:
        torch_device = torch.device(device) if isinstance(device, str) else device
    
    return _C.sparse_lu_factorize(sparse_matrix, torch_device)


def sparse_lu_solve(
    sparse_matrix: torch.Tensor, 
    rhs: torch.Tensor,
    factorize: bool = True,
    device: Optional[Union[torch.device, str]] = None
) -> torch.Tensor:
    """
    Solve a sparse linear system using PanguLU.
    
    Args:
        sparse_matrix: PyTorch sparse tensor in COO format (coefficient matrix)
        rhs: Right-hand side tensor
        factorize: Whether to perform factorization (True) or use pre-computed factors
        device: Device to perform computation on. If None, automatically detects from input tensors.
                For GPU tensors, uses GPU if CUDA support is available, otherwise falls back to CPU.
        
    Returns:
        Solution tensor x such that sparse_matrix @ x = rhs, on the same device as the input
        
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
    
    # Convert device to torch.device if string
    torch_device = None
    if device is not None:
        torch_device = torch.device(device) if isinstance(device, str) else device
    
    return _C.sparse_lu_solve(sparse_matrix, rhs, factorize, torch_device)