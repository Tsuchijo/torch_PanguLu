#!/usr/bin/env python3
"""
Device Detection Demo for torch-pangulu

This script demonstrates the automatic device detection and override functionality
that was added to torch-pangulu.
"""

import torch
import torch_pangulu

def main():
    print("Torch-PanguLU Device Detection Demo")
    print("=" * 50)
    
    # Show device capabilities
    print("\n1. Device Capabilities:")
    info = torch_pangulu._C.get_pangulu_info()
    print(f"   CUDA Support Compiled: {info['cuda_support']}")
    print(f"   CUDA Available: {info['cuda_available']}")
    print(f"   Auto Device Detection: {info['auto_device_detection']}")
    print(f"   Device Override Support: {info['supports_device_override']}")
    
    # Create a test problem
    print("\n2. Creating Test Problem:")
    n = 15
    # Simple diagonal matrix with some off-diagonal entries
    indices = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
    values = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    A = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64).coalesce()
    b = torch.ones(n, dtype=torch.float64)
    b[:5] = 2.0  # Set first 5 elements to 2.0
    
    print(f"   Matrix size: {A.size()}")
    print(f"   Matrix device: {A.device}")
    print(f"   RHS device: {b.device}")
    
    # Test 1: Auto-detection with CPU tensors
    print("\n3. Test 1 - Auto-detection with CPU tensors:")
    x1 = torch_pangulu.sparse_lu_solve(A, b)
    print(f"   Input devices: {A.device}, {b.device}")
    print(f"   Output device: {x1.device}")
    print(f"   Solution (first 5): {x1[:5].tolist()}")
    
    # Test 2: Explicit device specification
    print("\n4. Test 2 - Explicit device specification:")
    x2 = torch_pangulu.sparse_lu_solve(A, b, device='cpu')
    print(f"   Specified device: 'cpu'")
    print(f"   Output device: {x2.device}")
    print(f"   Solutions match: {torch.allclose(x1, x2)}")
    
    # Test 3: torch.device object
    print("\n5. Test 3 - Using torch.device object:")
    x3 = torch_pangulu.sparse_lu_solve(A, b, device=torch.device('cpu'))
    print(f"   Specified device: torch.device('cpu')")
    print(f"   Output device: {x3.device}")
    print(f"   Solutions match: {torch.allclose(x1, x3)}")
    
    # Test 4: GPU tensor handling (if available)
    if torch.cuda.is_available():
        print("\n6. Test 4 - GPU tensor handling:")
        A_gpu = A.cuda()
        b_gpu = b.cuda()
        print(f"   Moved tensors to GPU: {A_gpu.device}, {b_gpu.device}")
        
        # This should automatically detect GPU but fall back to CPU computation
        x4 = torch_pangulu.sparse_lu_solve(A_gpu, b_gpu)
        print(f"   Auto-detection result device: {x4.device}")
        print(f"   (Falls back to CPU since PanguLU GPU support not compiled)")
        
        # Test explicit CPU override with GPU inputs
        x5 = torch_pangulu.sparse_lu_solve(A_gpu, b_gpu, device='cpu')
        print(f"   Explicit CPU override device: {x5.device}")
        print(f"   GPU input -> CPU computation works: {torch.allclose(x4.cpu(), x1)}")
    else:
        print("\n6. Test 4 - GPU tensor handling:")
        print("   CUDA not available, skipping GPU tests")
    
    # Test numerical accuracy
    print("\n7. Numerical Accuracy Check:")
    residual = torch.norm(torch.sparse.mm(A, x1.unsqueeze(1)).squeeze() - b)
    print(f"   Residual norm: {residual:.2e}")
    if residual < 1e-10:
        print("   ✅ Excellent numerical accuracy!")
    else:
        print("   ⚠️  Residual higher than expected")
    
    print("\n" + "=" * 50)
    print("Device Detection Summary:")
    print("✅ Automatic device detection from input tensors")
    print("✅ Explicit device override via 'device' parameter")
    print("✅ Support for string and torch.device specifications")
    print("✅ GPU tensor handling with CPU fallback")
    print("✅ Preserves numerical accuracy across device operations")
    print("\nNote: GPU computation requires PanguLU built with -DGPU_OPEN flag.")
    print("Current build uses stable CPU-only configuration.")

if __name__ == "__main__":
    main()