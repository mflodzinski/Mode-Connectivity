#!/bin/bash

# Script to copy Fashion-MNIST training results from cluster to local
# Run this from the project root directory

CLUSTER="mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"


echo "Copying endpoints..."
scp ${CLUSTER}/results/convfc/fmnist/endpoints/standard/seed0/checkpoint-200.pt \
    results/convfc/fmnist/endpoints/standard/seed0/

scp ${CLUSTER}/results/convfc/fmnist/endpoints/standard/seed1/checkpoint-200.pt \
    results/convfc/fmnist/endpoints/standard/seed1/

