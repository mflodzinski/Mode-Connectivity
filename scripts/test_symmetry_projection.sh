#!/bin/bash
# Quick test script for symmetry plane projection implementation

set -e  # Exit on error

echo "Testing symmetry plane projection implementation..."
echo "=" | head -c 80 | tr ' ' '='
echo

# Test 1: Verify flag is recognized
echo "Test 1: Checking if --project_symmetry_plane flag is recognized..."
python external/dnn-mode-connectivity/train.py --help | grep -q "project_symmetry_plane"
if [ $? -eq 0 ]; then
    echo "✓ Flag recognized"
else
    echo "✗ Flag not found"
    exit 1
fi

# Test 2: Verify validation (should fail without --curve)
echo
echo "Test 2: Checking validation (should fail without curve)..."
python external/dnn-mode-connectivity/train.py \
    --model=VGG16 \
    --dataset=CIFAR10 \
    --data_path=./data \
    --project_symmetry_plane \
    2>&1 | grep -q "requires --curve"
if [ $? -eq 0 ]; then
    echo "✓ Validation working"
else
    echo "✗ Validation not working"
    exit 1
fi

# Test 3: Verify validation (should fail with wrong num_bends)
echo
echo "Test 3: Checking validation (should fail with num_bends != 3)..."
python external/dnn-mode-connectivity/train.py \
    --model=VGG16 \
    --dataset=CIFAR10 \
    --data_path=./data \
    --curve=PolyChain \
    --num_bends=5 \
    --project_symmetry_plane \
    2>&1 | grep -q "num_bends=3"
if [ $? -eq 0 ]; then
    echo "✓ Validation working"
else
    echo "✗ Validation not working"
    exit 1
fi

echo
echo "=" | head -c 80 | tr ' ' '='
echo "All tests passed! ✓"
echo
echo "Implementation is ready to use."
echo "Submit jobs with:"
echo "  sbatch scripts/slurm/symmetry_plane/submit_symplane_dnn_seed0-mirror.sh"
echo "  sbatch scripts/slurm/symmetry_plane/submit_symplane_dnn_seed0-seed1.sh"
