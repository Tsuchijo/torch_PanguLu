from setuptools import setup, Extension
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch
import os

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Get PyTorch include directories
torch_include_dirs = [
    f"{torch.utils.cmake_prefix_path}/include/torch/csrc/api/include",
    f"{torch.utils.cmake_prefix_path}/include",
]

# PanguLU paths (adjust these based on your PanguLU installation)
pangulu_include = os.environ.get('PANGULU_INCLUDE_DIR', './third_party/PanguLU/include')
pangulu_lib = os.environ.get('PANGULU_LIB_DIR', './third_party/PanguLU/lib')

# Compiler flags
cxx_flags = [
    '-O3',
    '-std=c++14',
    '-DWITH_PYTHON',
    '-DCALCULATE_TYPE_CR64',  # Complex double precision
]

# Libraries to link
libraries = ['torch', 'torch_python']
library_dirs = []

# Add CUDA support if available
if use_cuda:
    cxx_flags.append('-DUSE_CUDA')
    libraries.extend(['cudart', 'cublas', 'cusparse'])

# Add MPI support (required by PanguLU) - disabled for testing
# cxx_flags.append('-DUSE_MPI')
# libraries.append('mpi')

# Define the extension
ext_modules = [
    Pybind11Extension(
        "torch_pangulu._C",
        sources=[
            "src/torch_pangulu.cpp",
        ],
        include_dirs=[
            pangulu_include,
            *torch_include_dirs,
        ],
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=14,
        extra_compile_args=cxx_flags,
    ),
]

setup(
    name="torch-pangulu",
    version="0.1.0",
    author="Your Name",
    description="PyTorch C++ extension for PanguLU sparse LU decomposition",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=["torch_pangulu"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
)