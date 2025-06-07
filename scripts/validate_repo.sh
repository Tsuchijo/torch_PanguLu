#!/bin/bash

# Repository Validation Script
# Checks if the repository is ready for pushing and development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Check if we're in the right directory
check_project_root() {
    print_status "Checking project root directory..."
    
    if [ ! -f "setup.py" ] || [ ! -d "torch_pangulu" ]; then
        print_error "Not in torch-pangulu project root directory"
        return 1
    fi
    
    print_success "In correct project directory"
    return 0
}

# Check repository structure
check_structure() {
    print_status "Checking repository structure..."
    
    local required_files=(
        "README.md"
        "BUILD.md" 
        "DEVELOPMENT.md"
        "LICENSE"
        "setup.py"
        "CMakeLists.txt"
        ".gitignore"
        "torch_pangulu/__init__.py"
        "torch_pangulu/sparse_lu.py"
        "src/torch_pangulu.cpp"
        "tests/test_sparse_lu.py"
        "scripts/setup_pangulu.sh"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "All required files present"
    else
        print_error "Missing files: ${missing_files[*]}"
        return 1
    fi
    
    return 0
}

# Check that PanguLU is properly excluded
check_pangulu_exclusion() {
    print_status "Checking PanguLU exclusion..."
    
    if [ -d "third_party/PanguLU" ]; then
        print_success "PanguLU found in third_party/ directory"
    else
        print_warning "PanguLU not found - this is OK if not yet downloaded"
    fi
    
    # Check that PanguLU is in .gitignore
    if grep -q "third_party/" .gitignore && grep -q "PanguLU/" .gitignore; then
        print_success "PanguLU properly excluded in .gitignore"
    else
        print_error "PanguLU not properly excluded in .gitignore"
        return 1
    fi
    
    return 0
}

# Check for build artifacts that shouldn't be committed
check_build_artifacts() {
    print_status "Checking for uncommitted build artifacts..."
    
    local artifacts_found=0
    
    # Check for common build artifacts
    if find . -name "*.o" -o -name "*.a" -o -name "*.so" -o -name "CMakeCache.txt" | grep -q .; then
        print_warning "Found build artifacts - these should be cleaned before commit"
        find . -name "*.o" -o -name "*.a" -o -name "*.so" -o -name "CMakeCache.txt" | head -5
        artifacts_found=1
    fi
    
    if [ -d "build" ]; then
        print_warning "Found build/ directory - should be cleaned before commit"
        artifacts_found=1
    fi
    
    if [ $artifacts_found -eq 0 ]; then
        print_success "No build artifacts found"
    fi
    
    return 0
}

# Check Python package structure
check_python_package() {
    print_status "Checking Python package structure..."
    
    # Try to import the package (this will show warnings but shouldn't crash)
    if python3 -c "import torch_pangulu; print('Package imports successfully')" 2>/dev/null; then
        print_success "Python package structure is valid"
    else
        print_warning "Python package has import issues (may be normal without C++ extension)"
    fi
    
    return 0
}

# Check documentation
check_documentation() {
    print_status "Checking documentation completeness..."
    
    local doc_issues=0
    
    # Check README has installation instructions
    if ! grep -q "Installation" README.md; then
        print_error "README.md missing installation section"
        doc_issues=$((doc_issues + 1))
    fi
    
    # Check BUILD.md exists and has content
    if [ ! -s "BUILD.md" ]; then
        print_error "BUILD.md is empty or missing"
        doc_issues=$((doc_issues + 1))
    fi
    
    # Check DEVELOPMENT.md exists and has roadmap
    if ! grep -q "Development Status" DEVELOPMENT.md; then
        print_error "DEVELOPMENT.md missing development status"
        doc_issues=$((doc_issues + 1))
    fi
    
    if [ $doc_issues -eq 0 ]; then
        print_success "Documentation appears complete"
    else
        print_error "Found $doc_issues documentation issues"
        return 1
    fi
    
    return 0
}

# Check scripts are executable
check_scripts() {
    print_status "Checking script permissions..."
    
    local scripts_dir="scripts"
    if [ -d "$scripts_dir" ]; then
        local non_executable=()
        
        for script in "$scripts_dir"/*.sh; do
            if [ -f "$script" ] && [ ! -x "$script" ]; then
                non_executable+=("$script")
            fi
        done
        
        if [ ${#non_executable[@]} -eq 0 ]; then
            print_success "All scripts are executable"
        else
            print_warning "Non-executable scripts found: ${non_executable[*]}"
            print_status "Run: chmod +x ${non_executable[*]}"
        fi
    fi
    
    return 0
}

# Generate repository summary
generate_summary() {
    print_status "Generating repository summary..."
    
    echo ""
    echo "==============================================="
    echo "TORCH-PANGULU REPOSITORY VALIDATION SUMMARY"
    echo "==============================================="
    
    echo ""
    echo "Repository Statistics:"
    echo "- Total files: $(find . -type f | wc -l)"
    echo "- Python files: $(find . -name "*.py" | wc -l)"
    echo "- C++ files: $(find . -name "*.cpp" -o -name "*.h" | wc -l)"
    echo "- Documentation files: $(find . -name "*.md" | wc -l)"
    
    echo ""
    echo "Key Files:"
    ls -la README.md BUILD.md DEVELOPMENT.md LICENSE setup.py 2>/dev/null || true
    
    echo ""
    echo "Package Structure:"
    tree torch_pangulu/ 2>/dev/null || find torch_pangulu/ -type f | sort
    
    echo ""
    echo "Third-party Dependencies:"
    if [ -d "third_party" ]; then
        ls -la third_party/ 2>/dev/null || true
    else
        echo "  No third_party/ directory (will be created on setup)"
    fi
    
    echo ""
    echo "Ready for Development:"
    echo "✅ Repository structure complete"
    echo "✅ Documentation comprehensive"
    echo "✅ Build system configured"
    echo "✅ Third-party dependencies excluded"
    echo "⚠️  PanguLU integration incomplete (mock implementation)"
    echo "⚠️  CUDA support requires implementation"
    
    echo ""
    echo "Next Steps:"
    echo "1. Push repository to version control"
    echo "2. Clone on CUDA-enabled system"
    echo "3. Run: ./scripts/setup_pangulu.sh --cuda"
    echo "4. Complete PanguLU integration (see DEVELOPMENT.md)"
    echo "5. Implement CUDA support"
    echo "6. Run comprehensive tests"
    
    echo ""
    echo "==============================================="
}

# Main validation function
main() {
    echo "Starting torch-pangulu repository validation..."
    echo ""
    
    local failed_checks=0
    
    # Run all checks
    check_project_root || failed_checks=$((failed_checks + 1))
    check_structure || failed_checks=$((failed_checks + 1))
    check_pangulu_exclusion || failed_checks=$((failed_checks + 1))
    check_build_artifacts || failed_checks=$((failed_checks + 1))
    check_python_package || failed_checks=$((failed_checks + 1))
    check_documentation || failed_checks=$((failed_checks + 1))
    check_scripts || failed_checks=$((failed_checks + 1))
    
    echo ""
    
    if [ $failed_checks -eq 0 ]; then
        print_success "All validation checks passed!"
        generate_summary
        echo ""
        print_success "Repository is ready for pushing and development!"
        return 0
    else
        print_error "$failed_checks validation checks failed"
        echo ""
        print_error "Please fix the issues above before pushing"
        return 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo ""
        echo "Validates the torch-pangulu repository structure and readiness for development."
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "This script checks:"
        echo "  - Repository structure and required files"
        echo "  - PanguLU exclusion from git"
        echo "  - Build artifact cleanup"
        echo "  - Python package structure"
        echo "  - Documentation completeness"
        echo "  - Script permissions"
        exit 0
        ;;
    "")
        # No arguments - run validation
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac