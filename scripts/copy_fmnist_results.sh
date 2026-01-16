#!/bin/bash

# Script to copy Fashion-MNIST training results from cluster to local
# Run this from the project root directory

CLUSTER="mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"



echo "Copying linear evaluation..."
scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/convfc/fmnist/endpoints/standard/seed0-seed1/evaluations/linear.npz \
    results/convfc/fmnist/endpoints/standard/seed0-seed1/evaluations/ 2>/dev/null || \

mkdir -p results/convfc/fmnist/endpoints/standard/seed0-seed1/evaluations && \
scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/convfc/fmnist/endpoints/standard/seed0-seed1/evaluations/linear.npz \
    results/convfc/fmnist/endpoints/standard/seed0-seed1/evaluations/
