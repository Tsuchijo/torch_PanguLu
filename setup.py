from setuptools import setup, Extension
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch
import os

# Check if CUDA is available
# Can be overridden by environment variable TORCH_CUDA_FORCE
cuda_force = os.environ.get('TORCH_CUDA_FORCE', '').lower()
if cuda_force == '1' or cuda_force == 'true':
    use_cuda = True
elif cuda_force == '0' or cuda_force == 'false':
    use_cuda = False
else:
    use_cuda = torch.cuda.is_available()  # Auto-detect CUDA availability

# Get PyTorch include directories
torch_include_dirs = [
    os.path.join(os.path.dirname(torch.__file__), 'include'),
    os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'csrc', 'api', 'include'),
]

# PanguLU paths (adjust these based on your PanguLU installation)
pangulu_include = os.environ.get('PANGULU_INCLUDE_DIR', './third_party/PanguLU/include')
pangulu_lib = os.environ.get('PANGULU_LIB_DIR', './third_party/PanguLU/lib')

# Compiler flags
cxx_flags = [
    '-O3',
    '-std=c++17',
    '-DWITH_PYTHON',
    '-DCALCULATE_TYPE_R64',  # Real double precision (not complex)
]

# Libraries to link
libraries = ['torch', 'torch_python']
library_dirs = [
    os.path.join(os.path.dirname(torch.__file__), 'lib'),
    pangulu_lib,
    '/usr/lib/x86_64-linux-gnu/openblas-pthread',
    '/usr/lib/x86_64-linux-gnu/openmpi/lib',
] + (['/usr/lib/cuda/lib64', '/usr/lib/x86_64-linux-gnu'] if use_cuda else [])

# Add PanguLU library
libraries.extend(['pangulu'])

# Add MPI support (required by PanguLU) - use the correct OpenMPI library
libraries.extend(['mpi'])

# Add OpenBLAS
libraries.extend(['openblas'])

# Add threading support
libraries.extend(['pthread'])

# Add CUDA support if available
if use_cuda:
    cxx_flags.append('-DUSE_CUDA')
    libraries.extend(['cudart', 'cublas', 'cusparse'])

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
            '/usr/lib/x86_64-linux-gnu/openmpi/include',
            '/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi',
        ] + (['/usr/lib/cuda/include'] if use_cuda else []),
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
        extra_compile_args=cxx_flags,
        extra_link_args=[
            f'-Wl,-rpath,{pangulu_lib}',
            '-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib',
            '-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openblas-pthread',
        ] + (['-Wl,-rpath,/usr/lib/cuda/lib64', '-Wl,-rpath,/usr/lib/x86_64-linux-gnu'] if use_cuda else []),
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