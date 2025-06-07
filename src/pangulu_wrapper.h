#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

// PanguLU type definitions (from example.c)
typedef unsigned long long int sparse_pointer_t;
typedef unsigned int sparse_index_t;

// Default to double precision real numbers
#ifndef CALCULATE_TYPE_R64
#define CALCULATE_TYPE_R64
#endif

#if defined(CALCULATE_TYPE_R64)
typedef double sparse_value_t;
#elif defined(CALCULATE_TYPE_R32)
typedef float sparse_value_t;
#elif defined(CALCULATE_TYPE_CR64)
typedef double _Complex sparse_value_t;
#elif defined(CALCULATE_TYPE_CR32)
typedef float _Complex sparse_value_t;
#endif

extern "C" {
    #include "../third_party/PanguLU/include/pangulu.h"
}

namespace torch_pangulu {

/**
 * RAII wrapper for PanguLU matrix data structure
 */
class PanguLUMatrix {
public:
    PanguLUMatrix();
    ~PanguLUMatrix();
    
    // Non-copyable but movable
    PanguLUMatrix(const PanguLUMatrix&) = delete;
    PanguLUMatrix& operator=(const PanguLUMatrix&) = delete;
    PanguLUMatrix(PanguLUMatrix&&) = default;
    PanguLUMatrix& operator=(PanguLUMatrix&&) = default;
    
    // Initialize from PyTorch sparse tensor
    void from_torch_sparse(const torch::Tensor& sparse_tensor);
    
    // Convert back to PyTorch sparse tensors
    std::pair<torch::Tensor, torch::Tensor> to_torch_factors() const;
    
    // Perform LU factorization
    void factorize();
    
    // Solve linear system
    torch::Tensor solve(const torch::Tensor& rhs);
    
    // Check if factorization is done
    bool is_factorized() const { return factorized_; }
    
    // Get matrix dimensions
    int64_t rows() const { return rows_; }
    int64_t cols() const { return cols_; }
    
private:
    void* pangulu_handle_;
    pangulu_init_options init_options_;
    pangulu_gstrf_options gstrf_options_;
    pangulu_gstrs_options gstrs_options_;
    bool factorized_;
    int64_t rows_;
    int64_t cols_;
    
    void cleanup();
    void check_initialized() const;
};

/**
 * Utility functions for tensor conversion
 */
namespace utils {
    // Convert PyTorch sparse COO tensor to PanguLU CSR format
    void torch_sparse_to_csr(
        const torch::Tensor& sparse_tensor,
        std::vector<sparse_pointer_t>& row_ptr,
        std::vector<sparse_index_t>& col_indices,
        std::vector<sparse_value_t>& values
    );
    
    // Create PyTorch sparse tensor from CSR format
    torch::Tensor csr_to_torch_sparse(
        const std::vector<sparse_pointer_t>& row_ptr,
        const std::vector<sparse_index_t>& col_indices,
        const std::vector<sparse_value_t>& values,
        int64_t rows,
        int64_t cols,
        const torch::TensorOptions& options
    );
    
    // Validate sparse tensor format
    void validate_sparse_tensor(const torch::Tensor& tensor);
    
    // Check tensor compatibility for solve
    void check_solve_compatibility(
        const torch::Tensor& matrix,
        const torch::Tensor& rhs
    );
}

} // namespace torch_pangulu