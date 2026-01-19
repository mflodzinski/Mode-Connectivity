#!/bin/bash

echo "Checking endpoint distances for VGG16/CIFAR10..."
poetry run python scripts/analysis/check_endpoint_distances.py \
    --model VGG16 \
    --dataset CIFAR10 \
    --checkpoint1 results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt \
    --checkpoint2 results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
    --checkpoint3 results/vgg16/cifar10/endpoints/standard/seed2/checkpoints/checkpoint-200.pt \
    --seed-names seed0 seed1 seed2

echo ""
echo ""
echo "Checking endpoint distances for ConvFC/FashionMNIST..."
poetry run python scripts/analysis/check_endpoint_distances.py \
    --model ConvFC \
    --dataset FashionMNIST \
    --checkpoint1 results/convfc/fmnist/endpoints/standard/seed0/checkpoints/checkpoint-200.pt \
    --checkpoint2 results/convfc/fmnist/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
    --checkpoint3 results/convfc/fmnist/endpoints/standard/seed2/checkpoints/checkpoint-200.pt \
    --seed-names seed0 seed1 seed2
