cmake_minimum_required(VERSION 3.12)
project(torch_pangulu)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

# Optional packages
find_package(CUDA QUIET)
find_package(OpenMP)

# PanguLU paths
set(PANGULU_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/PanguLU" CACHE PATH "PanguLU root directory")
set(PANGULU_INCLUDE_DIR "${PANGULU_ROOT_DIR}/include" CACHE PATH "PanguLU include directory")
set(PANGULU_LIB_DIR "${PANGULU_ROOT_DIR}/lib" CACHE PATH "PanguLU library directory")

# Check if PanguLU library exists (optional for testing)
find_library(PANGULU_LIB pangulu PATHS ${PANGULU_LIB_DIR} NO_DEFAULT_PATH)
if(NOT PANGULU_LIB)
    message(WARNING "PanguLU library not found. Building test version without PanguLU integration.")
    set(PANGULU_AVAILABLE FALSE)
else()
    set(PANGULU_AVAILABLE TRUE)
endif()

# Compiler definitions
add_definitions(-DCALCULATE_TYPE_R64)  # Real double precision
add_definitions(-DUSE_MPI)

if(CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
endif()

if(OpenMP_FOUND)
    add_definitions(-DUSE_OPENMP)
endif()

# Include directories
include_directories(
    ${PANGULU_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
)

# Source files
set(SOURCES
    src/torch_pangulu.cpp
)

# Create the extension module
add_library(torch_pangulu_cpp SHARED ${SOURCES})

# Link libraries
target_link_libraries(torch_pangulu_cpp 
    ${TORCH_LIBRARIES}
)

if(PANGULU_AVAILABLE)
    target_link_libraries(torch_pangulu_cpp ${PANGULU_LIB})
endif()

if(MPI_FOUND)
    target_link_libraries(torch_pangulu_cpp ${MPI_LIBRARIES})
endif()

if(CUDA_FOUND)
    target_link_libraries(torch_pangulu_cpp 
        ${CUDA_LIBRARIES}
        cublas
        cusparse
    )
endif()

if(OpenMP_FOUND)
    target_link_libraries(torch_pangulu_cpp OpenMP::OpenMP_CXX)
endif()

# Compiler flags
target_compile_options(torch_pangulu_cpp PRIVATE 
    ${TORCH_CXX_FLAGS}
    ${MPI_COMPILE_FLAGS}
    -O3
    -fPIC
)

# Set properties
set_property(TARGET torch_pangulu_cpp PROPERTY CXX_STANDARD 14)

# Installation
install(TARGETS torch_pangulu_cpp DESTINATION lib)