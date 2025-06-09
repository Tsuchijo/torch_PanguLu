#!/usr/bin/env python3
"""
Basic usage example for torch-pangulu sparse LU decomposition.

This example demonstrates how to:
1. Create a sparse matrix
2. Solve a linear system using PanguLU
3. Verify the solution
"""

import torch
import numpy as np
import torch_pangulu
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import spsolve


def create_test_matrix(n=1000, density=0.01):
    """Create a test sparse matrix using scipy and convert to PyTorch."""
    # Create a random sparse matrix
    scipy_matrix = sparse_random(n, n, density=density, format='coo', dtype=np.float64)
    
    # Make it symmetric positive definite for stability
    scipy_matrix = scipy_matrix + scipy_matrix.T
    scipy_matrix.setdiag(scipy_matrix.diagonal() + n * 0.1)
    
    # Ensure it's in COO format
    scipy_matrix = scipy_matrix.tocoo()
    
    # Convert to PyTorch sparse tensor
    indices = torch.from_numpy(np.vstack([scipy_matrix.row, scipy_matrix.col])).long()
    values = torch.from_numpy(scipy_matrix.data).double()
    shape = scipy_matrix.shape
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    return sparse_tensor, scipy_matrix


def main():
    print("Torch-PanguLU Basic Usage Example")
    print("=" * 40)
    
    # Check PanguLU availability
    try:
        info = torch_pangulu._C.get_pangulu_info()
        print(f"PanguLU Info: {info}")
    except Exception as e:
        print(f"Warning: Could not get PanguLU info: {e}")
    
    # Create test problem
    print("\n1. Creating test sparse matrix...")
    n = 500
    sparse_matrix, scipy_matrix = create_test_matrix(n, density=0.02)
    
    print(f"   Matrix size: {sparse_matrix.shape}")
    print(f"   Non-zeros: {sparse_matrix._nnz()}")
    print(f"   Density: {sparse_matrix._nnz() / (n * n) * 100:.2f}%")
    
    # Create right-hand side
    print("\n2. Creating right-hand side vector...")
    x_true = torch.randn(n, dtype=torch.float64)
    b = torch.sparse.mm(sparse_matrix, x_true.unsqueeze(1)).squeeze()
    
    print(f"   RHS norm: {torch.norm(b):.6f}")
    
    # Solve using PanguLU
    print("\n3. Solving using PanguLU...")
    try:
        x_pangulu = torch_pangulu.sparse_lu_solve(sparse_matrix, b)
        
        # Compute residual
        residual = torch.sparse.mm(sparse_matrix, x_pangulu.unsqueeze(1)).squeeze() - b
        residual_norm = torch.norm(residual)
        solution_error = torch.norm(x_pangulu - x_true)
        
        print(f"   Solution computed successfully!")
        print(f"   Residual norm: {residual_norm:.2e}")
        print(f"   Solution error: {solution_error:.2e}")
        
    except Exception as e:
        print(f"   PanguLU solve failed: {e}")
        print("   This is expected if PanguLU is not properly installed")
    
    # Compare with scipy (reference solution)
    print("\n4. Reference solution using SciPy...")
    try:
        x_scipy = spsolve(scipy_matrix.tocsr(), b.numpy())
        x_scipy = torch.from_numpy(x_scipy).double()
        
        scipy_error = torch.norm(x_scipy - x_true)
        print(f"   SciPy solution error: {scipy_error:.2e}")
        
        # Compare solutions if both succeeded
        try:
            difference = torch.norm(x_pangulu - x_scipy)
            print(f"   PanguLU vs SciPy difference: {difference:.2e}")
        except:
            pass
            
    except Exception as e:
        print(f"   SciPy solve failed: {e}")
    
    # Test factorization (if available)
    print("\n5. Testing factorization interface...")
    try:
        L, U = torch_pangulu.sparse_lu_factorize(sparse_matrix)
        print(f"   Factorization completed!")
        print(f"   L shape: {L.shape}, U shape: {U.shape}")
    except Exception as e:
        print(f"   Factorization failed: {e}")
        print("   Factor extraction may not be implemented yet")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()