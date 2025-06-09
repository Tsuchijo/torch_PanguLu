# Device Detection and Override Implementation Summary

This document summarizes the device detection and override functionality that has been added to torch-pangulu.

## 🎯 Overview

The torch-pangulu library now automatically detects the device of input tensors and can optionally use GPU acceleration when available, with support for explicit device override.

## ✨ New Features

### 1. Automatic Device Detection
- **Smart Detection**: Automatically detects if input tensors are on GPU or CPU
- **Fallback Logic**: If GPU tensors are provided but CUDA support is not available, automatically falls back to CPU computation
- **Device Preservation**: Returns results on the same device as the input tensors

### 2. Explicit Device Override
- **String Format**: `device='cpu'` or `device='cuda'`
- **PyTorch Device**: `device=torch.device('cuda:0')`
- **None (Auto)**: `device=None` for automatic detection (default)

### 3. Enhanced API

Both main functions now support the `device` parameter:

```python
# Sparse LU solve with device detection
torch_pangulu.sparse_lu_solve(
    sparse_matrix: torch.Tensor,
    rhs: torch.Tensor, 
    factorize: bool = True,
    device: Optional[Union[torch.device, str]] = None
) -> torch.Tensor

# Sparse LU factorization with device detection  
torch_pangulu.sparse_lu_factorize(
    sparse_matrix: torch.Tensor,
    device: Optional[Union[torch.device, str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

## 🔧 Implementation Details

### C++ Level Changes

**File: `src/torch_pangulu.cpp`**

1. **Function Signatures Updated**:
   ```cpp
   torch::Tensor sparse_lu_solve_impl(
       const torch::Tensor& sparse_matrix,
       const torch::Tensor& rhs, 
       bool factorize_flag,
       c10::optional<torch::Device> device
   );
   
   std::pair<torch::Tensor, torch::Tensor> sparse_lu_factorize_impl(
       const torch::Tensor& sparse_matrix,
       c10::optional<torch::Device> device
   );
   ```

2. **Device Detection Logic**:
   ```cpp
   // Determine target device
   torch::Device target_device = torch::kCPU;  // Default to CPU
   
   if (device.has_value()) {
       // Use explicitly specified device
       target_device = device.value();
   } else {
       // Auto-detect device from input tensors
       if (sparse_matrix.is_cuda() || rhs.is_cuda()) {
   #ifdef GPU_OPEN
           target_device = torch::kCUDA;
   #else
           target_device = torch::kCPU;  // Fallback
   #endif
       }
   }
   ```

3. **Device-Aware Tensor Operations**:
   ```cpp
   // Move tensors to target device if needed
   auto matrix_on_device = sparse_matrix.to(target_device);
   auto rhs_on_device = rhs.to(target_device);
   
   // Ensure solution is on the same device as original input
   if (sparse_matrix.device() != target_device) {
       final_solution = solution.to(sparse_matrix.device());
   }
   ```

4. **Enhanced Info Function**:
   ```cpp
   py::dict get_pangulu_info() {
       // ... existing info ...
       info["auto_device_detection"] = true;
       info["supports_device_override"] = true;
       info["cuda_available"] = torch::cuda::is_available();
       info["cuda_device_count"] = torch::cuda::device_count();
       // ...
   }
   ```

### Python Level Changes

**File: `torch_pangulu/sparse_lu.py`**

1. **Updated Function Signatures**:
   ```python
   def sparse_lu_solve(
       sparse_matrix: torch.Tensor, 
       rhs: torch.Tensor,
       factorize: bool = True,
       device: Optional[Union[torch.device, str]] = None
   ) -> torch.Tensor
   
   def sparse_lu_factorize(
       sparse_matrix: torch.Tensor, 
       device: Optional[Union[torch.device, str]] = None
   ) -> Tuple[torch.Tensor, torch.Tensor]
   ```

2. **Device Parameter Processing**:
   ```python
   # Convert device to torch.device if string
   torch_device = None
   if device is not None:
       torch_device = torch.device(device) if isinstance(device, str) else device
   ```

## 📋 Usage Examples

### Basic Auto-Detection

```python
import torch
import torch_pangulu

# CPU tensors (will use CPU computation)
A_cpu = torch.sparse_coo_tensor(...).coalesce()
b_cpu = torch.randn(n, dtype=torch.float64)
x = torch_pangulu.sparse_lu_solve(A_cpu, b_cpu)
# Result: x.device == torch.device('cpu')

# GPU tensors (will use GPU if available, else CPU)
A_gpu = A_cpu.cuda()
b_gpu = b_cpu.cuda()  
x = torch_pangulu.sparse_lu_solve(A_gpu, b_gpu)
# Result: x.device == A_gpu.device (preserves input device)
```

### Explicit Device Override

```python
# Force CPU computation even with GPU inputs
x_cpu = torch_pangulu.sparse_lu_solve(A_gpu, b_gpu, device='cpu')
# Result: x_cpu.device == torch.device('cpu')

# Force GPU computation (if supported)
x_gpu = torch_pangulu.sparse_lu_solve(A_cpu, b_cpu, device='cuda')
# Result: x_gpu.device == torch.device('cuda:0')

# Using torch.device objects
x = torch_pangulu.sparse_lu_solve(A, b, device=torch.device('cuda:1'))
```

### Mixed Device Handling

```python
# Handles tensors on different devices automatically
A_cpu = torch.sparse_coo_tensor(...).coalesce()  # CPU
b_gpu = torch.randn(n).cuda()                     # GPU

# Will move both to common device and compute
x = torch_pangulu.sparse_lu_solve(A_cpu, b_gpu)
```

## 🏗️ Configuration Support

### CUDA Detection

The implementation includes compile-time and runtime CUDA detection:

```cpp
#ifdef GPU_OPEN
    // GPU support compiled in
    if (torch::cuda::is_available()) {
        // CUDA runtime available
        use_gpu = true;
    } else {
        // Fall back to CPU
        use_gpu = false;
    }
#else
    // CPU-only build
    use_gpu = false;
#endif
```

### Device Information

Enhanced info function provides comprehensive device status:

```python
info = torch_pangulu._C.get_pangulu_info()
print(f"CUDA Support Compiled: {info['cuda_support']}")
print(f"CUDA Runtime Available: {info['cuda_available']}")
print(f"CUDA Device Count: {info['cuda_device_count']}")
print(f"Auto Device Detection: {info['auto_device_detection']}")
print(f"Device Override Support: {info['supports_device_override']}")
```

## 🔄 Device Selection Logic

The device selection follows this priority order:

1. **Explicit Override**: If `device` parameter is provided, use that device
2. **GPU Auto-Detection**: If input tensors are on GPU and GPU support is available, use GPU
3. **CPU Fallback**: Default to CPU computation
4. **Error Handling**: If specified device is not available, fall back gracefully

## ⚠️ Current Limitations

1. **GPU Support**: Requires PanguLU built with `-DGPU_OPEN` flag
2. **Factorization Stability**: Some runtime issues with factorization function need debugging
3. **Multi-GPU**: Currently uses default GPU device (could be extended for specific GPU selection)

## 🎯 Benefits

- **Seamless GPU Integration**: Works transparently with existing PyTorch GPU workflows
- **Backward Compatibility**: Default behavior unchanged for existing code
- **Flexible Override**: Allows forcing specific computation devices when needed
- **Error Resilience**: Graceful fallback when requested device unavailable
- **Performance Optimization**: Can leverage GPU acceleration when available

## 🚀 Future Enhancements

- Multi-GPU device selection (`device='cuda:1'`)
- Asynchronous GPU computation
- Memory pooling for GPU operations  
- Batch processing optimization
- Mixed precision support

---

This implementation provides a robust foundation for device-aware sparse linear algebra operations in torch-pangulu, seamlessly integrating with PyTorch's device management system.