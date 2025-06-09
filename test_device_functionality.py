#!/usr/bin/env python3
"""
Simple test for device detection functionality.
"""

import torch
import torch_pangulu

def test_device_detection():
    """Test basic device detection functionality."""
    print("Testing Device Detection Functionality")
    print("=" * 50)
    
    # Create simple test matrix
    n = 10
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    values = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)
    A = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64).coalesce()
    b = torch.ones(n, dtype=torch.float64)
    b[:3] = 3.0
    
    # Test 1: Original API (without device parameter)
    print("Test 1: Original API")
    try:
        x1 = torch_pangulu.sparse_lu_solve(A, b)
        residual1 = torch.norm(torch.sparse.mm(A, x1.unsqueeze(1)).squeeze() - b)
        print(f"  ✅ Original API works, residual: {residual1:.2e}")
    except Exception as e:
        print(f"  ❌ Original API failed: {e}")
        return False
    
    # Test 2: Device parameter with None
    print("Test 2: Device parameter with None")
    try:
        x2 = torch_pangulu.sparse_lu_solve(A, b, device=None)
        if torch.allclose(x1, x2):
            print("  ✅ device=None gives same result as original API")
        else:
            print("  ❌ device=None gives different result")
            return False
    except Exception as e:
        print(f"  ❌ device=None failed: {e}")
        return False
    
    # Test 3: Explicit CPU device
    print("Test 3: Explicit CPU device")
    try:
        x3 = torch_pangulu.sparse_lu_solve(A, b, device='cpu')
        if torch.allclose(x1, x3):
            print("  ✅ device='cpu' gives same result")
        else:
            print("  ❌ device='cpu' gives different result")
            return False
    except Exception as e:
        print(f"  ❌ device='cpu' failed: {e}")
        return False
    
    # Test 4: torch.device object
    print("Test 4: torch.device object")
    try:
        x4 = torch_pangulu.sparse_lu_solve(A, b, device=torch.device('cpu'))
        if torch.allclose(x1, x4):
            print("  ✅ device=torch.device('cpu') gives same result")
        else:
            print("  ❌ device=torch.device('cpu') gives different result")
            return False
    except Exception as e:
        print(f"  ❌ device=torch.device('cpu') failed: {e}")
        return False
    
    # Test 5: Factorization function
    print("Test 5: Factorization function")
    try:
        L, U = torch_pangulu.sparse_lu_factorize(A)
        print(f"  ✅ Factorization works, L shape: {L.shape}, U shape: {U.shape}")
        
        # Test with device parameter
        L2, U2 = torch_pangulu.sparse_lu_factorize(A, device='cpu')
        print(f"  ✅ Factorization with device parameter works")
    except Exception as e:
        print(f"  ❌ Factorization failed: {e}")
        return False
    
    # Test 6: Enhanced info function
    print("Test 6: Enhanced info function")
    try:
        info = torch_pangulu._C.get_pangulu_info()
        required_keys = ['available', 'cuda_support', 'auto_device_detection', 'supports_device_override']
        for key in required_keys:
            if key not in info:
                print(f"  ❌ Missing key in info: {key}")
                return False
        print(f"  ✅ Info function has all required keys")
        print(f"  ✅ Auto device detection: {info['auto_device_detection']}")
        print(f"  ✅ Device override support: {info['supports_device_override']}")
    except Exception as e:
        print(f"  ❌ Info function failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All device detection tests PASSED!")
    print("✅ Original functionality preserved")
    print("✅ New device features working")
    print("✅ Backward compatibility maintained")
    return True

if __name__ == "__main__":
    success = test_device_detection()
    exit(0 if success else 1)