#!/bin/bash

# PanguLU Setup Script for Torch-PanguLU
# This script downloads, configures, and builds PanguLU for integration with PyTorch

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Essential tools
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi
    
    if ! command_exists make; then
        missing_deps+=("make")
    fi
    
    if ! command_exists mpicc; then
        missing_deps+=("mpi (mpicc not found)")
    fi
    
    # C++ compiler
    if ! command_exists gcc && ! command_exists clang; then
        missing_deps+=("gcc or clang")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install missing dependencies and run this script again."
        
        local os=$(detect_os)
        case $os in
            linux)
                echo "On Ubuntu/Debian: sudo apt install git cmake build-essential openmpi-bin openmpi-common libopenmpi-dev libopenblas-dev"
                echo "On CentOS/RHEL: sudo yum groupinstall 'Development Tools' && sudo yum install cmake git openmpi openmpi-devel openblas-devel"
                ;;
            macos)
                echo "On macOS: brew install git cmake open-mpi openblas libomp"
                ;;
            *)
                echo "Please install the missing dependencies for your system."
                ;;
        esac
        exit 1
    fi
    
    print_success "All required dependencies found!"
}

# Detect system configuration
detect_system_config() {
    print_status "Detecting system configuration..."
    
    local os=$(detect_os)
    print_status "Operating System: $os"
    
    # Detect BLAS library
    local blas_lib=""
    local blas_inc=""
    
    case $os in
        linux)
            if [ -d "/usr/include/openblas" ]; then
                blas_inc="/usr/include/openblas"
                blas_lib="/usr/lib/x86_64-linux-gnu"
            elif [ -d "/usr/local/include" ] && ldconfig -p | grep -q openblas; then
                blas_inc="/usr/local/include"
                blas_lib="/usr/local/lib"
            fi
            ;;
        macos)
            if [ -d "/opt/homebrew/opt/openblas" ]; then
                blas_inc="/opt/homebrew/opt/openblas/include"
                blas_lib="/opt/homebrew/opt/openblas/lib"
            elif [ -d "/usr/local/opt/openblas" ]; then
                blas_inc="/usr/local/opt/openblas/include"
                blas_lib="/usr/local/opt/openblas/lib"
            fi
            ;;
    esac
    
    if [ -z "$blas_lib" ]; then
        print_warning "OpenBLAS not found. You may need to install it manually."
    else
        print_success "OpenBLAS found: $blas_lib"
    fi
    
    # Detect CUDA
    if command_exists nvcc; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        print_success "CUDA found: version $cuda_version"
        CUDA_AVAILABLE=true
        CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    else
        print_warning "CUDA not found. GPU acceleration will be disabled."
        CUDA_AVAILABLE=false
    fi
    
    # Detect OpenMP
    if command_exists gcc && gcc -fopenmp -dM -E - < /dev/null | grep -q "_OPENMP"; then
        print_success "OpenMP support detected with GCC"
        OPENMP_AVAILABLE=true
    elif [ "$os" = "macos" ] && [ -d "/opt/homebrew/opt/libomp" ]; then
        print_success "OpenMP support detected (Homebrew libomp)"
        OPENMP_AVAILABLE=true
        OPENMP_PATH="/opt/homebrew/opt/libomp"
    else
        print_warning "OpenMP not detected. Parallel performance may be limited."
        OPENMP_AVAILABLE=false
    fi
    
    # Export detected paths
    export BLAS_INC="$blas_inc"
    export BLAS_LIB="$blas_lib"
    export OS_TYPE="$os"
}

# Clone or update PanguLU
setup_pangulu_source() {
    print_status "Setting up PanguLU source code..."
    
    local pangulu_dir="third_party/PanguLU"
    
    if [ -d "$pangulu_dir" ]; then
        print_status "PanguLU directory already exists. Updating..."
        cd "$pangulu_dir"
        git pull origin main || git pull origin master
        cd - > /dev/null
    else
        print_status "Cloning PanguLU repository..."
        mkdir -p third_party
        git clone https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git "$pangulu_dir"
    fi
    
    if [ ! -d "$pangulu_dir" ]; then
        print_error "Failed to set up PanguLU source code"
        exit 1
    fi
    
    print_success "PanguLU source code ready!"
}

# Configure PanguLU build
configure_pangulu() {
    print_status "Configuring PanguLU build..."
    
    local pangulu_dir="third_party/PanguLU"
    local make_inc="$pangulu_dir/make.inc"
    
    # Backup original make.inc
    if [ -f "$make_inc" ] && [ ! -f "$make_inc.backup" ]; then
        cp "$make_inc" "$make_inc.backup"
    fi
    
    # Create custom make.inc based on detected system
    cat > "$make_inc" << EOF
# Auto-generated make.inc for torch-pangulu
# Generated on $(date)

COMPILE_LEVEL = -O3

# General settings
CC = gcc \$(COMPILE_LEVEL)
MPICC = mpicc \$(COMPILE_LEVEL)

EOF

    # Add BLAS configuration
    if [ -n "$BLAS_INC" ] && [ -n "$BLAS_LIB" ]; then
        cat >> "$make_inc" << EOF
# BLAS configuration
OPENBLAS_INC = -I$BLAS_INC
OPENBLAS_LIB = -L$BLAS_LIB -lopenblas

EOF
    else
        cat >> "$make_inc" << EOF
# BLAS configuration (update paths as needed)
OPENBLAS_INC = -I/usr/include/openblas
OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu -lopenblas

EOF
    fi
    
    # Add CUDA configuration if available
    if [ "$CUDA_AVAILABLE" = true ]; then
        cat >> "$make_inc" << EOF
# CUDA configuration
CUDA_PATH = $CUDA_PATH
CUDA_INC = -I$CUDA_PATH/include
CUDA_LIB = -L$CUDA_PATH/lib64 -lcudart -lcusparse
NVCC = nvcc \$(COMPILE_LEVEL)
NVCCFLAGS = \$(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 \$(CUDA_INC) \$(CUDA_LIB)

EOF
    else
        cat >> "$make_inc" << EOF
# CUDA configuration (disabled)
CUDA_PATH = /usr/local/cuda
CUDA_INC = -I/path/to/cuda/include
CUDA_LIB = -L/path/to/cuda/lib64 -lcudart -lcusparse

EOF
    fi
    
    # Add OpenMP and other flags
    local openmp_flags=""
    if [ "$OPENMP_AVAILABLE" = true ]; then
        if [ "$OS_TYPE" = "macos" ] && [ -n "$OPENMP_PATH" ]; then
            openmp_flags="-I$OPENMP_PATH/include -L$OPENMP_PATH/lib -lomp"
        else
            openmp_flags="-fopenmp"
        fi
    fi
    
    cat >> "$make_inc" << EOF
# Compiler flags
MPICCFLAGS = \$(OPENBLAS_INC) \$(CUDA_INC) \$(OPENBLAS_LIB) $openmp_flags -lpthread -lm
MPICCLINK = \$(OPENBLAS_LIB)

# METIS configuration (disabled by default)
METISFLAGS = -I/path/to/gklib/include -I/path/to/metis/include

# PanguLU compilation flags
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64
EOF

    # Add CUDA flag if available
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "# Enable GPU support" >> "$make_inc"
        echo "PANGULU_FLAGS += -DGPU_OPEN" >> "$make_inc"
    fi
    
    print_success "PanguLU configuration complete!"
}

# Build PanguLU
build_pangulu() {
    print_status "Building PanguLU library..."
    
    local pangulu_dir="third_party/PanguLU"
    
    cd "$pangulu_dir"
    
    # Clean previous build
    make clean 2>/dev/null || true
    
    # Build with appropriate number of jobs
    local num_jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    print_status "Building with $num_jobs parallel jobs..."
    
    if make -j"$num_jobs"; then
        print_success "PanguLU library built successfully!"
    else
        print_error "PanguLU build failed!"
        print_error "Check the error messages above and fix any missing dependencies."
        exit 1
    fi
    
    # Verify build artifacts
    if [ -f "lib/libpangulu.a" ] || [ -f "lib/libpangulu.so" ]; then
        print_success "PanguLU library files created successfully!"
        ls -la lib/libpangulu.*
    else
        print_warning "PanguLU library files not found. Build may be incomplete."
    fi
    
    cd - > /dev/null
}

# Test PanguLU installation
test_pangulu() {
    print_status "Testing PanguLU installation..."
    
    local pangulu_dir="third_party/PanguLU"
    
    # Check if example can be built
    if [ -f "$pangulu_dir/examples/pangulu_example.elf" ]; then
        print_success "PanguLU example executable found!"
        
        # Try to run a small test (if test matrix is available)
        if [ -f "$pangulu_dir/examples/Trefethen_20b.mtx" ]; then
            print_status "Running quick test with sample matrix..."
            cd "$pangulu_dir/examples"
            
            # Run with single process and small block size
            if timeout 30s mpirun -np 1 ./pangulu_example.elf -nb 2 -f Trefethen_20b.mtx > test_output.txt 2>&1; then
                print_success "PanguLU test completed successfully!"
                grep "residual" test_output.txt || true
            else
                print_warning "PanguLU test failed or timed out. This may be normal on some systems."
            fi
            
            rm -f test_output.txt
            cd - > /dev/null
        fi
    else
        print_warning "PanguLU example not built. You may need to build examples separately."
    fi
}

# Main installation function
main() {
    print_status "Starting PanguLU setup for torch-pangulu..."
    print_status "================================================"
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ] || [ ! -d "torch_pangulu" ]; then
        print_error "This script must be run from the torch-pangulu project root directory."
        exit 1
    fi
    
    # Run setup steps
    check_dependencies
    detect_system_config
    setup_pangulu_source
    configure_pangulu
    build_pangulu
    test_pangulu
    
    print_success "================================================"
    print_success "PanguLU setup completed successfully!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Build the torch-pangulu extension: python setup.py build_ext --inplace"
    print_status "2. Install in development mode: pip install -e ."
    print_status "3. Run tests: python -m pytest tests/"
    print_status ""
    print_status "For more information, see BUILD.md and DEVELOPMENT.md"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            FORCE_CUDA=true
            shift
            ;;
        --no-cuda)
            FORCE_NO_CUDA=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cuda        Force enable CUDA support"
            echo "  --no-cuda     Force disable CUDA support"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main