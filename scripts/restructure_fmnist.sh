#!/bin/bash

# Script to restructure convfc/fmnist results to match vgg16/cifar10 structure
# This adds missing checkpoints/ subdirectories

echo "Restructuring convfc/fmnist results..."

# Function to move file to checkpoints subdirectory
move_to_checkpoints() {
    local file=$1
    local dir=$(dirname "$file")
    local filename=$(basename "$file")

    if [ -f "$file" ]; then
        echo "Moving $file to $dir/checkpoints/$filename"
        mkdir -p "$dir/checkpoints"
        mv "$file" "$dir/checkpoints/$filename"
    fi
}

# Move endpoint checkpoints
move_to_checkpoints "results/convfc/fmnist/endpoints/standard/seed0/checkpoint-200.pt"
move_to_checkpoints "results/convfc/fmnist/endpoints/standard/seed1/checkpoint-200.pt"

# Move polygon checkpoint
move_to_checkpoints "results/convfc/fmnist/advanced_geometry/polygon/seed0-seed1/checkpoint-150.pt"

# Move symmetry plane checkpoint
move_to_checkpoints "results/convfc/fmnist/advanced_geometry/symmetry_plane/seed0-seed1/checkpoint-150.pt"

echo "Done! Structure now matches vgg16/cifar10"
echo ""
echo "New structure:"
find results/convfc/fmnist -type f \( -name "*.pt" -o -name "*.npz" \) | sort
