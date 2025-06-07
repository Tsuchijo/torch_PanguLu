#include "pangulu_wrapper.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace torch_pangulu {

PanguLUMatrix::PanguLUMatrix() 
    : pangulu_handle_(nullptr), factorized_(false), rows_(0), cols_(0) {
    // Initialize PanguLU options with default values
    init_options_.nb = 4;        // Default block size
    init_options_.nthread = 20;  // Default thread count
}

PanguLUMatrix::~PanguLUMatrix() {
    cleanup();
}

void PanguLUMatrix::cleanup() {
    if (pangulu_handle_) {
        pangulu_finalize(&pangulu_handle_);
        pangulu_handle_ = nullptr;
    }
    factorized_ = false;
}

void PanguLUMatrix::check_initialized() const {
    if (!pangulu_handle_) {
        throw std::runtime_error("PanguLU matrix not properly initialized");
    }
}

void PanguLUMatrix::from_torch_sparse(const torch::Tensor& sparse_tensor) {
    utils::validate_sparse_tensor(sparse_tensor);
    
    // Get tensor dimensions
    rows_ = sparse_tensor.size(0);
    cols_ = sparse_tensor.size(1);
    
    // Convert to CSR format
    std::vector<sparse_pointer_t> row_ptr;
    std::vector<sparse_index_t> col_indices;
    std::vector<sparse_value_t> values;
    utils::torch_sparse_to_csr(sparse_tensor, row_ptr, col_indices, values);
    
    // Initialize PanguLU solver
    sparse_index_t n = static_cast<sparse_index_t>(rows_);
    sparse_pointer_t nnz = static_cast<sparse_pointer_t>(values.size());
    
    pangulu_init(n, nnz, row_ptr.data(), col_indices.data(), values.data(), 
                 &init_options_, &pangulu_handle_);
    
    factorized_ = false;
}

void PanguLUMatrix::factorize() {
    check_initialized();
    
    pangulu_gstrf(&gstrf_options_, &pangulu_handle_);
    factorized_ = true;
}

torch::Tensor PanguLUMatrix::solve(const torch::Tensor& rhs) {
    check_initialized();
    
    if (!factorized_) {
        factorize();
    }
    
    // Validate RHS dimensions
    if (rhs.size(0) != rows_) {
        throw std::runtime_error("RHS dimension mismatch");
    }
    
    // Prepare solution tensor (copy from rhs as PanguLU modifies in-place)
    auto solution = rhs.clone().contiguous();
    
    // Convert to appropriate data type for double precision
    solution = solution.to(torch::kFloat64);
    
    sparse_value_t* sol_data = solution.data_ptr<sparse_value_t>();
    
    // Solve using PanguLU (modifies sol_data in-place)
    pangulu_gstrs(sol_data, &gstrs_options_, &pangulu_handle_);
    
    return solution.to(rhs.dtype());
}

std::pair<torch::Tensor, torch::Tensor> PanguLUMatrix::to_torch_factors() const {
    check_initialized();
    
    if (!factorized_) {
        throw std::runtime_error("Matrix must be factorized before extracting factors");
    }
    
    // This is a simplified implementation
    // In practice, you'd need to extract L and U factors from PanguLU's internal structure
    // The exact implementation depends on PanguLU's API for accessing factors
    
    throw std::runtime_error("Factor extraction not yet implemented - depends on PanguLU internal API");
}

namespace utils {

void torch_sparse_to_csr(
    const torch::Tensor& sparse_tensor,
    std::vector<sparse_pointer_t>& row_ptr,
    std::vector<sparse_index_t>& col_indices,
    std::vector<sparse_value_t>& values) {
    
    // Get COO format data
    auto indices = sparse_tensor.indices();
    auto vals = sparse_tensor.values();
    
    int64_t rows = sparse_tensor.size(0);
    int64_t nnz = vals.size(0);
    
    // Extract row and column indices
    auto row_indices_acc = indices[0].accessor<int64_t, 1>();
    auto col_indices_acc = indices[1].accessor<int64_t, 1>();
    auto values_acc = vals.accessor<double, 1>();
    
    // Convert to CSR format
    row_ptr.resize(rows + 1, 0);
    col_indices.resize(nnz);
    values.resize(nnz);
    
    // Count entries per row
    for (int64_t i = 0; i < nnz; i++) {
        row_ptr[row_indices_acc[i] + 1]++;
    }
    
    // Convert counts to pointers
    for (int64_t i = 1; i <= rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }
    
    // Fill column indices and values
    std::vector<sparse_pointer_t> row_counters(rows, 0);
    for (int64_t i = 0; i < nnz; i++) {
        int64_t row = row_indices_acc[i];
        sparse_pointer_t pos = row_ptr[row] + row_counters[row];
        col_indices[pos] = static_cast<sparse_index_t>(col_indices_acc[i]);
        values[pos] = values_acc[i];
        row_counters[row]++;
    }
}

torch::Tensor csr_to_torch_sparse(
    const std::vector<sparse_pointer_t>& row_ptr,
    const std::vector<sparse_index_t>& col_indices,
    const std::vector<sparse_value_t>& values,
    int64_t rows,
    int64_t cols,
    const torch::TensorOptions& options) {
    
    int64_t nnz = values.size();
    
    // Create indices tensor
    auto indices = torch::zeros({2, nnz}, torch::kLong);
    auto indices_acc = indices.accessor<int64_t, 2>();
    
    // Convert CSR to COO
    int64_t idx = 0;
    for (int64_t row = 0; row < rows; row++) {
        for (sparse_pointer_t ptr = row_ptr[row]; ptr < row_ptr[row + 1]; ptr++) {
            indices_acc[0][idx] = row;
            indices_acc[1][idx] = col_indices[ptr];
            idx++;
        }
    }
    
    // Create values tensor  
    auto vals = torch::from_blob(
        const_cast<sparse_value_t*>(values.data()), 
        {nnz}, 
        torch::kFloat64
    ).clone().to(options.dtype());
    
    return torch::sparse_coo_tensor(indices, vals, {rows, cols}, options);
}

void validate_sparse_tensor(const torch::Tensor& tensor) {
    if (!tensor.is_sparse()) {
        throw std::invalid_argument("Input must be a sparse tensor");
    }
    
    if (tensor.layout() != torch::kSparseCoo) {
        throw std::invalid_argument("Sparse tensor must be in COO format");
    }
    
    if (tensor.dim() != 2) {
        throw std::invalid_argument("Sparse tensor must be 2-dimensional");
    }
    
    if (tensor.size(0) != tensor.size(1)) {
        throw std::invalid_argument("Matrix must be square for LU decomposition");
    }
}

void check_solve_compatibility(
    const torch::Tensor& matrix,
    const torch::Tensor& rhs) {
    
    validate_sparse_tensor(matrix);
    
    if (rhs.dim() < 1 || rhs.dim() > 2) {
        throw std::invalid_argument("RHS must be 1D or 2D tensor");
    }
    
    if (matrix.size(0) != rhs.size(0)) {
        throw std::invalid_argument("Matrix rows must match RHS rows");
    }
}

} // namespace utils

} // namespace torch_pangulu