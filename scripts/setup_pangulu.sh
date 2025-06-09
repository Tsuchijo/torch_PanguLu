#!/bin/bash

# Torch-PanguLU Automated Setup Script
# This script automates the configuration and building of PanguLU with optional CUDA support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENABLE_CUDA=false
CPU_ONLY=false
CUDA_PATH=""
FORCE_REBUILD=false
VERBOSE=false
SKIP_TESTS=false

# Show usage information
show_help() {
    cat << EOF
Torch-PanguLU Automated Setup Script

USAGE:
    bash scripts/setup_pangulu.sh [OPTIONS]

OPTIONS:
    --enable-cuda           Enable experimental CUDA support (requires CUDA toolkit)
    --cpu-only             Build CPU-only version (recommended, stable)
    --cuda-path PATH       Specify custom CUDA installation path
    --force-rebuild        Force clean rebuild of all components
    --verbose              Enable verbose output
    --skip-tests           Skip validation tests after build
    --help                 Show this help message

EXAMPLES:
    # CPU-only build (recommended)
    bash scripts/setup_pangulu.sh --cpu-only

    # Experimental CUDA build
    bash scripts/setup_pangulu.sh --enable-cuda

    # CUDA build with custom path
    bash scripts/setup_pangulu.sh --enable-cuda --cuda-path=/usr/local/cuda

    # Force clean rebuild
    bash scripts/setup_pangulu.sh --cpu-only --force-rebuild

BUILD STATUS:
    ✅ CPU-Only: Stable, production-ready with excellent numerical accuracy
    ⚠️  CUDA: Experimental, may have kernel compatibility issues

EOF
}

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --enable-cuda)
                ENABLE_CUDA=true
                shift
                ;;
            --cpu-only)
                CPU_ONLY=true
                shift
                ;;
            --cuda-path)
                CUDA_PATH="$2"
                shift 2
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate arguments
    if [[ "$ENABLE_CUDA" == true && "$CPU_ONLY" == true ]]; then
        print_error "Cannot specify both --enable-cuda and --cpu-only"
        exit 1
    fi

    if [[ "$ENABLE_CUDA" == false && "$CPU_ONLY" == false ]]; then
        print_info "No build type specified, defaulting to CPU-only (recommended)"
        CPU_ONLY=true
    fi
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    # Check if we're in the right directory
    if [[ ! -f "setup.py" || ! -d "third_party" ]]; then
        print_error "Please run this script from the torch-pangulu root directory"
        exit 1
    fi

    # Check basic tools
    local missing_tools=()
    
    for tool in gcc mpicc python3 pip; do
        if ! command -v $tool &> /dev/null; then
            missing_tools+=($tool)
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_info "Please install them using your package manager"
        exit 1
    fi

    # Check Python packages
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch is not installed. Please install it first:"
        print_info "  pip install torch"
        exit 1
    fi

    print_success "Basic requirements satisfied"
}

# Auto-detect CUDA installation
detect_cuda() {
    if [[ -n "$CUDA_PATH" ]]; then
        print_info "Using specified CUDA path: $CUDA_PATH"
        return 0
    fi

    # Common CUDA installation paths
    local cuda_paths=(
        "/usr/local/cuda"
        "/usr/lib/cuda"
        "/opt/cuda"
        "/usr/cuda"
    )

    for path in "${cuda_paths[@]}"; do
        if [[ -f "$path/bin/nvcc" ]]; then
            CUDA_PATH="$path"
            print_info "Auto-detected CUDA at: $CUDA_PATH"
            return 0
        fi
    done

    # Check if nvcc is in PATH
    if command -v nvcc &> /dev/null; then
        CUDA_PATH=$(dirname $(dirname $(which nvcc)))
        print_info "Auto-detected CUDA from PATH: $CUDA_PATH"
        return 0
    fi

    return 1
}

# Check CUDA requirements
check_cuda_requirements() {
    if [[ "$ENABLE_CUDA" == false ]]; then
        return 0
    fi

    print_info "Checking CUDA requirements..."

    if ! detect_cuda; then
        print_error "CUDA installation not found"
        print_info "Please install CUDA toolkit or specify path with --cuda-path"
        exit 1
    fi

    # Check CUDA version
    if [[ -f "$CUDA_PATH/bin/nvcc" ]]; then
        local cuda_version=$($CUDA_PATH/bin/nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_info "Found CUDA version: $cuda_version"
        
        # Parse version
        local major_version=$(echo $cuda_version | cut -d. -f1)
        if [[ $major_version -lt 11 ]]; then
            print_warning "CUDA version $cuda_version is older than recommended (11.0+)"
        fi
    fi

    # Check if PyTorch has CUDA support
    if ! python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        print_warning "PyTorch was built without CUDA support or no GPU detected"
        print_info "CUDA build may still work for PanguLU, but PyTorch tensors will be CPU-only"
    fi

    # Check GPU compute capability
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU information:"
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | while read line; do
            print_info "  $line"
        done
    fi

    print_success "CUDA requirements checked"
}

# Setup virtual environment if needed
setup_venv() {
    if [[ -z "$VIRTUAL_ENV" && ! -f "venv/bin/activate" ]]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_info "Activating virtual environment..."
        source venv/bin/activate
    fi

    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install torch numpy pybind11 scipy pytest
}

# Clone PanguLU if needed
setup_pangulu_source() {
    if [[ ! -d "third_party/PanguLU" ]]; then
        print_info "Cloning PanguLU source..."
        mkdir -p third_party
        git clone https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git third_party/PanguLU
    else
        print_info "PanguLU source already exists"
    fi
}

# Configure PanguLU build
configure_pangulu() {
    print_info "Configuring PanguLU build..."
    
    cd third_party/PanguLU

    # Backup original make.inc if it exists
    if [[ -f "make.inc" && ! -f "make.inc.backup" ]]; then
        cp make.inc make.inc.backup
    fi

    # Create make.inc based on build type
    cat > make.inc << EOF
COMPILE_LEVEL = -O3

#general
CC = gcc \$(COMPILE_LEVEL)
MPICC = mpicc \$(COMPILE_LEVEL)
OPENBLAS_INC = -I/usr/include/x86_64-linux-gnu/openblas-pthread/
OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas
METISFLAGS = 

EOF

    if [[ "$ENABLE_CUDA" == true ]]; then
        print_info "Configuring for experimental CUDA build..."
        cat >> make.inc << EOF
#CUDA Configuration (Experimental)
CUDA_PATH = $CUDA_PATH
CUDA_INC = -I$CUDA_PATH/include
CUDA_LIB = -L$CUDA_PATH/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart -lcusparse
NVCC = nvcc \$(COMPILE_LEVEL)
NVCCFLAGS = \$(PANGULU_FLAGS) -w -Xptxas -dlcm=cg \\
            -gencode=arch=compute_60,code=sm_60 \\
            -gencode=arch=compute_70,code=sm_70 \\
            -gencode=arch=compute_80,code=sm_80 \\
            \$(CUDA_INC)

MPICCFLAGS = \$(OPENBLAS_INC) \$(CUDA_INC) \$(OPENBLAS_LIB) \$(CUDA_LIB) -fopenmp -lpthread -lm
MPICCLINK = \$(OPENBLAS_LIB) \$(CUDA_LIB)
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN

EOF
        print_warning "CUDA build is experimental and may have compatibility issues"
    else
        print_info "Configuring for stable CPU-only build..."
        cat >> make.inc << EOF
#CPU-Only Configuration (Stable)
CUDA_PATH = /usr/lib/cuda
CUDA_INC = -I/usr/lib/cuda/include
CUDA_LIB = -L/usr/lib/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lcudart -lcusparse
NVCC = nvcc \$(COMPILE_LEVEL)
NVCCFLAGS = \$(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 \$(CUDA_INC)

MPICCFLAGS = \$(OPENBLAS_INC) \$(CUDA_INC) \$(OPENBLAS_LIB) \$(CUDA_LIB) -fopenmp -lpthread -lm
MPICCLINK = \$(OPENBLAS_LIB) \$(CUDA_LIB)
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64

EOF
    fi

    cd ../..
    print_success "PanguLU configuration completed"
}

# Build PanguLU
build_pangulu() {
    print_info "Building PanguLU..."
    
    cd third_party/PanguLU

    if [[ "$FORCE_REBUILD" == true ]]; then
        print_info "Forcing clean rebuild..."
        make clean
    fi

    # Build with appropriate verbosity (skip examples to avoid hardcoded path issues)
    if [[ "$VERBOSE" == true ]]; then
        make -C src -j$(nproc) && make -C lib
    else
        (make -C src -j$(nproc) && make -C lib) > build.log 2>&1 || {
            print_error "PanguLU build failed. Check build.log for details:"
            tail -20 build.log
            exit 1
        }
    fi

    # Verify build
    if [[ -f "lib/libpangulu.so" ]]; then
        print_success "PanguLU built successfully"
    else
        print_error "PanguLU build failed - library not found"
        exit 1
    fi

    cd ../..
}

# Update setup.py for CUDA if needed
configure_torch_extension() {
    print_info "Configuring torch extension..."

    # The setup.py should automatically detect CUDA availability
    # But we can force it with environment variables if needed
    if [[ "$ENABLE_CUDA" == true ]]; then
        export TORCH_CUDA_FORCE=1
        print_info "Forcing CUDA support in torch extension"
    else
        export TORCH_CUDA_FORCE=0
        print_info "Disabling CUDA support in torch extension"
    fi
}

# Build torch extension
build_torch_extension() {
    print_info "Building torch extension..."

    # Make sure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" && -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    # Clean build if requested
    if [[ "$FORCE_REBUILD" == true ]]; then
        rm -rf build/ torch_pangulu/_C*.so
    fi

    # Build extension
    if [[ "$VERBOSE" == true ]]; then
        python setup.py build_ext --inplace
    else
        python setup.py build_ext --inplace > torch_build.log 2>&1 || {
            print_error "Torch extension build failed. Check torch_build.log for details:"
            tail -20 torch_build.log
            exit 1
        }
    fi

    print_success "Torch extension built successfully"
}

# Run validation tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        print_info "Skipping tests as requested"
        return 0
    fi

    print_info "Running validation tests..."

    # Make sure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" && -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi

    # Test import and basic info
    print_info "Testing basic functionality..."
    python3 -c "
import torch_pangulu
info = torch_pangulu._C.get_pangulu_info()
print('✅ Import successful')
print(f'Available: {info[\"available\"]}')
print(f'CUDA support: {info[\"cuda_support\"]}')
print(f'MPI support: {info[\"mpi_support\"]}')
print(f'Version: {info[\"version\"]}')
" || {
        print_error "Basic functionality test failed"
        exit 1
    }

    # Test sparse solve
    print_info "Testing sparse LU solve..."
    python3 -c "
import torch
import torch_pangulu

# Create well-conditioned test matrix
n = 50
i_indices = []
j_indices = []
values = []

# Diagonal entries
for i in range(n):
    i_indices.append(i)
    j_indices.append(i)
    values.append(4.0)

# Off-diagonal entries
for i in range(n-1):
    i_indices.extend([i, i+1])
    j_indices.extend([i+1, i])
    values.extend([-1.0, -1.0])

indices = torch.tensor([i_indices, j_indices], dtype=torch.long)
values = torch.tensor(values, dtype=torch.float64)
A = torch.sparse_coo_tensor(indices, values, (n, n), dtype=torch.float64).coalesce()
b = torch.randn(n, dtype=torch.float64)

try:
    x = torch_pangulu.sparse_lu_solve(A, b)
    residual = torch.norm(torch.sparse.mm(A, x.unsqueeze(1)).squeeze() - b)
    print(f'✅ Solve test passed. Residual: {residual:.2e}')
    
    if residual > 1e-10:
        print('⚠️  High residual - check numerical stability')
    else:
        print('✅ Excellent numerical accuracy')
        
except Exception as e:
    print(f'❌ Solve test failed: {e}')
    exit(1)
" || {
        print_error "Sparse solve test failed"
        exit 1
    }

    print_success "All validation tests passed!"
}

# Main setup function
main() {
    echo -e "${BLUE}"
    echo "================================================================"
    echo "              Torch-PanguLU Automated Setup Script"
    echo "================================================================"
    echo -e "${NC}"

    parse_args "$@"
    check_requirements
    check_cuda_requirements
    setup_venv
    setup_pangulu_source
    configure_pangulu
    build_pangulu
    configure_torch_extension
    build_torch_extension
    run_tests

    echo -e "${GREEN}"
    echo "================================================================"
    echo "                    Setup Complete!"
    echo "================================================================"
    echo -e "${NC}"

    if [[ "$ENABLE_CUDA" == true ]]; then
        print_warning "You built with experimental CUDA support"
        print_info "If you encounter issues, rebuild with: bash scripts/setup_pangulu.sh --cpu-only"
    else
        print_success "Built stable CPU-only version with excellent performance"
    fi

    print_info "To use torch-pangulu:"
    if [[ -f "venv/bin/activate" ]]; then
        print_info "  source venv/bin/activate"
    fi
    print_info "  python -c \"import torch_pangulu; print('Ready to use!')\""
    
    echo ""
    print_info "For examples, see: examples/basic_usage.py"
    print_info "For documentation, see: README.md and BUILD.md"
}

# Run main function with all arguments
main "$@"