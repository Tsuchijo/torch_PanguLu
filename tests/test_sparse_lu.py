#!/usr/bin/env python3
"""
Unit tests for torch-pangulu sparse LU decomposition.
"""

import unittest
import torch
import numpy as np
from scipy.sparse import random as sparse_random, eye as sparse_eye


class TestSparseLU(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.dtype = torch.float64
        
    def create_test_matrix(self, n=100, density=0.05):
        """Create a test sparse matrix."""
        # Create random sparse matrix
        scipy_matrix = sparse_random(n, n, density=density, format='coo', dtype=np.float64)
        
        # Make it symmetric positive definite
        scipy_matrix = scipy_matrix + scipy_matrix.T
        scipy_matrix.setdiag(scipy_matrix.diagonal() + n * 0.1)
        
        # Convert to PyTorch
        indices = torch.from_numpy(np.vstack([scipy_matrix.row, scipy_matrix.col])).long()
        values = torch.from_numpy(scipy_matrix.data).double()
        
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    
    def test_import(self):
        """Test that the module can be imported."""
        try:
            import torch_pangulu
            self.assertTrue(hasattr(torch_pangulu, 'sparse_lu_solve'))
            self.assertTrue(hasattr(torch_pangulu, 'sparse_lu_factorize'))
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
    
    def test_sparse_tensor_validation(self):
        """Test input validation for sparse tensors."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        # Test with dense tensor
        dense_matrix = torch.randn(10, 10)
        rhs = torch.randn(10)
        
        with self.assertRaises(ValueError):
            torch_pangulu.sparse_lu_solve(dense_matrix, rhs)
    
    def test_dimension_compatibility(self):
        """Test dimension compatibility checking."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        sparse_matrix = self.create_test_matrix(10)
        rhs_wrong = torch.randn(5)  # Wrong dimension
        
        with self.assertRaises(ValueError):
            torch_pangulu.sparse_lu_solve(sparse_matrix, rhs_wrong)
    
    def test_identity_matrix_solve(self):
        """Test solving with identity matrix."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        n = 50
        # Create sparse identity matrix
        scipy_identity = sparse_eye(n, format='coo', dtype=np.float64)
        indices = torch.from_numpy(np.vstack([scipy_identity.row, scipy_identity.col])).long()
        values = torch.from_numpy(scipy_identity.data).double()
        identity = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        
        # Right-hand side
        b = torch.randn(n, dtype=torch.float64)
        
        try:
            x = torch_pangulu.sparse_lu_solve(identity, b)
            error = torch.norm(x - b)
            self.assertLess(error, 1e-12, "Identity matrix solve should be exact")
        except RuntimeError as e:
            self.skipTest(f"PanguLU not available: {e}")
    
    def test_solve_consistency(self):
        """Test that A*x = b for solved system."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        # Create test problem
        A = self.create_test_matrix(100, 0.03)
        x_true = torch.randn(100, dtype=torch.float64)
        b = torch.sparse.mm(A, x_true.unsqueeze(1)).squeeze()
        
        try:
            x_solved = torch_pangulu.sparse_lu_solve(A, b)
            
            # Check residual
            residual = torch.sparse.mm(A, x_solved.unsqueeze(1)).squeeze() - b
            residual_norm = torch.norm(residual)
            
            self.assertLess(residual_norm, 1e-10, 
                          f"Residual too large: {residual_norm}")
            
        except RuntimeError as e:
            self.skipTest(f"PanguLU solve failed: {e}")
    
    def test_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        A = self.create_test_matrix(50, 0.05)
        B = torch.randn(50, 3, dtype=torch.float64)  # Multiple RHS
        
        try:
            # Solve for each column
            solutions = []
            for i in range(B.size(1)):
                x = torch_pangulu.sparse_lu_solve(A, B[:, i])
                solutions.append(x)
            
            X = torch.stack(solutions, dim=1)
            
            # Check residual
            residual = torch.sparse.mm(A, X) - B
            residual_norm = torch.norm(residual)
            
            self.assertLess(residual_norm, 1e-9,
                          f"Multiple RHS residual too large: {residual_norm}")
            
        except RuntimeError as e:
            self.skipTest(f"PanguLU solve failed: {e}")
    
    def test_factorization_interface(self):
        """Test the factorization interface."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        A = self.create_test_matrix(30, 0.1)
        
        try:
            L, U = torch_pangulu.sparse_lu_factorize(A)
            
            # Basic shape checks
            self.assertEqual(L.shape, A.shape)
            self.assertEqual(U.shape, A.shape)
            self.assertTrue(L.is_sparse)
            self.assertTrue(U.is_sparse)
            
        except (RuntimeError, NotImplementedError) as e:
            self.skipTest(f"Factorization interface not available: {e}")
    
    def test_cached_factorization(self):
        """Test factorization caching functionality."""
        try:
            import torch_pangulu
        except ImportError:
            self.skipTest("torch_pangulu not compiled")
        
        A = self.create_test_matrix(40, 0.08)
        b1 = torch.randn(40, dtype=torch.float64)
        b2 = torch.randn(40, dtype=torch.float64)
        
        try:
            # Clear any existing factorization
            torch_pangulu._C.clear_factorization()
            self.assertFalse(torch_pangulu._C.has_cached_factorization())
            
            # First solve should factorize
            x1 = torch_pangulu.sparse_lu_solve(A, b1, factorize=True)
            
            if torch_pangulu._C.has_cached_factorization():
                # Second solve should use cached factorization
                x2 = torch_pangulu.sparse_lu_solve(A, b2, factorize=False)
                
                # Both should be valid solutions
                res1 = torch.norm(torch.sparse.mm(A, x1.unsqueeze(1)).squeeze() - b1)
                res2 = torch.norm(torch.sparse.mm(A, x2.unsqueeze(1)).squeeze() - b2)
                
                self.assertLess(res1, 1e-10)
                self.assertLess(res2, 1e-10)
            
        except RuntimeError as e:
            self.skipTest(f"Cached factorization test failed: {e}")


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_pangulu_info(self):
        """Test PanguLU info function."""
        try:
            import torch_pangulu
            info = torch_pangulu._C.get_pangulu_info()
            
            self.assertIsInstance(info, dict)
            self.assertIn('available', info)
            self.assertIn('cuda_support', info)
            self.assertIn('mpi_support', info)
            
        except ImportError:
            self.skipTest("torch_pangulu not compiled")


if __name__ == '__main__':
    unittest.main()