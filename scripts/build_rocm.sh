#!/bin/bash
# ==============================================================================
# PyFlame ROCm Build Script
# ==============================================================================
# This script builds PyFlame with ROCm backend support for AMD GPUs.
#
# Usage:
#   ./scripts/build_rocm.sh [options]
#
# Options:
#   --clean          Clean build directory before building
#   --debug          Build in debug mode (default: Release)
#   --tests          Build and run tests
#   --benchmark      Build and run benchmarks
#   --install        Install after building
#   --prefix=PATH    Installation prefix (default: /usr/local)
#   --rocm=PATH      ROCm installation path (default: /opt/rocm)
#   --jobs=N         Number of parallel jobs (default: nproc)
#   --help           Show this help message
#
# Environment Variables:
#   ROCM_PATH        ROCm installation path (overridden by --rocm)
#   HIP_PATH         HIP installation path (default: $ROCM_PATH/hip)
#   MIOPEN_PATH      MIOpen installation path (default: $ROCM_PATH)
#
# Example:
#   ./scripts/build_rocm.sh --clean --tests
#   ./scripts/build_rocm.sh --rocm=/opt/rocm-5.7.0 --install
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Default Configuration
# ==============================================================================

BUILD_TYPE="Release"
BUILD_DIR="build-rocm"
INSTALL_PREFIX="/usr/local"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS=$(nproc)
DO_CLEAN=false
DO_TESTS=false
DO_BENCHMARK=false
DO_INSTALL=false

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_step() {
    echo -e "${GREEN}>>> $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# SECURITY: Validate path for dangerous characters
validate_path() {
    local path="$1"
    local name="$2"

    # Check for null bytes
    if [[ "$path" == *$'\0'* ]]; then
        print_error "$name contains null bytes"
        exit 1
    fi

    # Check for path traversal attempts
    if [[ "$path" == *".."* ]]; then
        print_error "$name contains path traversal sequence '..'"
        exit 1
    fi

    # Check for shell metacharacters that could cause issues
    if [[ "$path" =~ [[:cntrl:]] ]]; then
        print_error "$name contains control characters"
        exit 1
    fi

    # Check path doesn't start with - (could be interpreted as option)
    if [[ "$path" == -* ]] && [[ "$path" != /* ]]; then
        print_error "$name starts with '-' which could be interpreted as a command option"
        exit 1
    fi
}

# SECURITY: Validate numeric input
validate_numeric() {
    local value="$1"
    local name="$2"

    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        print_error "$name must be a positive integer, got: $value"
        exit 1
    fi

    # Sanity check for reasonable values
    if [[ "$value" -lt 1 ]] || [[ "$value" -gt 1024 ]]; then
        print_error "$name must be between 1 and 1024, got: $value"
        exit 1
    fi
}

show_help() {
    head -40 "$0" | tail -35 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

check_rocm() {
    print_step "Checking ROCm installation..."

    if [ ! -d "$ROCM_PATH" ]; then
        print_error "ROCm not found at $ROCM_PATH"
        echo "Please install ROCm or specify the path with --rocm=PATH"
        exit 1
    fi

    # Check for required components
    local missing=""

    if [ ! -f "$ROCM_PATH/bin/hipcc" ]; then
        missing="$missing hipcc"
    fi

    if [ ! -d "$ROCM_PATH/include/rocblas" ]; then
        missing="$missing rocblas"
    fi

    if [ ! -d "$ROCM_PATH/include/miopen" ]; then
        missing="$missing miopen"
    fi

    if [ -n "$missing" ]; then
        print_error "Missing ROCm components:$missing"
        echo "Please ensure ROCm is fully installed."
        exit 1
    fi

    # Get ROCm version
    if [ -f "$ROCM_PATH/.info/version" ]; then
        ROCM_VERSION=$(cat "$ROCM_PATH/.info/version")
        echo "  ROCm version: $ROCM_VERSION"
    elif [ -f "$ROCM_PATH/bin/rocm_agent_enumerator" ]; then
        ROCM_VERSION=$("$ROCM_PATH/bin/rocminfo" 2>/dev/null | grep -m1 "HSA Runtime" | awk '{print $NF}' || echo "unknown")
        echo "  ROCm version: $ROCM_VERSION"
    fi

    echo "  ROCm path: $ROCM_PATH"
    echo "  HIP compiler: $ROCM_PATH/bin/hipcc"

    print_step "ROCm installation OK"
}

check_gpu() {
    print_step "Checking AMD GPU..."

    if [ -x "$ROCM_PATH/bin/rocm-smi" ]; then
        echo ""
        "$ROCM_PATH/bin/rocm-smi" --showproductname 2>/dev/null || true
        echo ""
    else
        print_warning "rocm-smi not found, skipping GPU detection"
    fi
}

check_dependencies() {
    print_step "Checking build dependencies..."

    local missing=""

    # Check for CMake
    if ! command -v cmake &> /dev/null; then
        missing="$missing cmake"
    else
        CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
        echo "  CMake: $CMAKE_VERSION"
    fi

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        missing="$missing python3"
    else
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        echo "  Python: $PYTHON_VERSION"
    fi

    # Check for pip
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        missing="$missing pip"
    fi

    # Check for make or ninja
    if command -v ninja &> /dev/null; then
        BUILD_TOOL="ninja"
        echo "  Build tool: Ninja"
    elif command -v make &> /dev/null; then
        BUILD_TOOL="make"
        echo "  Build tool: Make"
    else
        missing="$missing make/ninja"
    fi

    if [ -n "$missing" ]; then
        print_error "Missing dependencies:$missing"
        echo "Please install the missing dependencies."
        exit 1
    fi

    print_step "Dependencies OK"
}

# ==============================================================================
# Parse Command Line Arguments
# ==============================================================================

for arg in "$@"; do
    case $arg in
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --tests)
            DO_TESTS=true
            shift
            ;;
        --benchmark)
            DO_BENCHMARK=true
            shift
            ;;
        --install)
            DO_INSTALL=true
            shift
            ;;
        --prefix=*)
            INSTALL_PREFIX="${arg#*=}"
            # SECURITY: Validate path input
            validate_path "$INSTALL_PREFIX" "Install prefix"
            shift
            ;;
        --rocm=*)
            ROCM_PATH="${arg#*=}"
            # SECURITY: Validate path input
            validate_path "$ROCM_PATH" "ROCm path"
            shift
            ;;
        --jobs=*)
            JOBS="${arg#*=}"
            # SECURITY: Validate numeric input
            validate_numeric "$JOBS" "Jobs"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $arg"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# SECURITY: Validate default paths as well
validate_path "$ROCM_PATH" "ROCm path"
validate_path "$INSTALL_PREFIX" "Install prefix"
validate_path "$BUILD_DIR" "Build directory"

# ==============================================================================
# Main Build Process
# ==============================================================================

print_header "PyFlame ROCm Build"

echo "Configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Build directory: $BUILD_DIR"
echo "  Parallel jobs: $JOBS"
echo "  Install prefix: $INSTALL_PREFIX"
echo ""

# Check prerequisites
check_rocm
check_gpu
check_dependencies

# Get the script's directory (PyFlame root)
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYFLAME_ROOT="$(dirname "$SCRIPT_DIR")"
cd -- "$PYFLAME_ROOT"

echo ""
echo "Building in: $PYFLAME_ROOT"
echo ""

# Clean if requested
if [ "$DO_CLEAN" = true ]; then
    print_step "Cleaning build directory..."
    # SECURITY: Use -- to prevent path from being interpreted as option
    rm -rf -- "$BUILD_DIR"
fi

# Create build directory
print_step "Creating build directory..."
# SECURITY: Use -- to prevent path from being interpreted as option
mkdir -p -- "$BUILD_DIR"
cd -- "$BUILD_DIR"

# Configure with CMake
print_header "Configuring with CMake"

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    "-DPYFLAME_USE_ROCM=ON"
    "-DCMAKE_PREFIX_PATH=$ROCM_PATH"
    "-DROCM_PATH=$ROCM_PATH"
    "-DHIP_PATH=${HIP_PATH:-$ROCM_PATH}"
)

# Use Ninja if available
if [ "$BUILD_TOOL" = "ninja" ]; then
    CMAKE_ARGS+=("-GNinja")
fi

# Enable tests if requested
if [ "$DO_TESTS" = true ]; then
    CMAKE_ARGS+=("-DPYFLAME_BUILD_TESTS=ON")
fi

echo "CMake arguments:"
for arg in "${CMAKE_ARGS[@]}"; do
    echo "  $arg"
done
echo ""

cmake "${CMAKE_ARGS[@]}" ..

# Build
print_header "Building PyFlame"

if [ "$BUILD_TOOL" = "ninja" ]; then
    ninja -j"$JOBS"
else
    make -j"$JOBS"
fi

print_step "Build complete!"

# Run tests if requested
if [ "$DO_TESTS" = true ]; then
    print_header "Running Tests"

    # C++ tests
    if [ -f "tests/pyflame_rocm_tests" ]; then
        print_step "Running C++ ROCm tests..."
        ./tests/pyflame_rocm_tests --gtest_color=yes
    else
        print_warning "C++ ROCm tests not found"
    fi

    # Python tests
    cd -- "$PYFLAME_ROOT"
    if [ -f "tests/python/test_rocm_backend.py" ]; then
        print_step "Running Python ROCm tests..."

        # Install Python package in development mode
        pip3 install -e python/ --quiet 2>/dev/null || true

        python3 -m pytest tests/python/test_rocm_backend.py -v --color=yes
    else
        print_warning "Python ROCm tests not found"
    fi

    cd -- "$PYFLAME_ROOT/$BUILD_DIR"
fi

# Run benchmarks if requested
if [ "$DO_BENCHMARK" = true ]; then
    print_header "Running Benchmarks"

    cd -- "$PYFLAME_ROOT"

    # Install Python package if needed
    pip3 install -e python/ --quiet 2>/dev/null || true

    if [ -f "benchmarks/rocm_benchmark.py" ]; then
        print_step "Running ROCm benchmarks..."
        python3 benchmarks/rocm_benchmark.py --all
    else
        print_warning "ROCm benchmarks not found"
    fi

    cd -- "$PYFLAME_ROOT/$BUILD_DIR"
fi

# Install if requested
if [ "$DO_INSTALL" = true ]; then
    print_header "Installing PyFlame"

    if [ "$BUILD_TOOL" = "ninja" ]; then
        ninja install
    else
        make install
    fi

    # Install Python package
    cd -- "$PYFLAME_ROOT"
    pip3 install -- python/

    print_step "Installation complete!"
    echo "  C++ library installed to: $INSTALL_PREFIX"
    echo "  Python package installed to site-packages"
fi

# Summary
print_header "Build Summary"

echo "Build type: $BUILD_TYPE"
echo "Build directory: $PYFLAME_ROOT/$BUILD_DIR"
echo ""

if [ -f "libpyflame.a" ] || [ -f "libpyflame.so" ]; then
    echo "Built libraries:"
    ls -lh libpyflame* 2>/dev/null || true
fi

echo ""
echo "Next steps:"
echo "  1. Install Python package: pip install -e python/"
echo "  2. Test ROCm support: python -c \"import pyflame; print(pyflame.rocm_is_available())\""
echo "  3. Run benchmarks: python benchmarks/rocm_benchmark.py --all"
echo ""

print_step "Done!"
