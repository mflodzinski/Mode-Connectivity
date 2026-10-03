# Frozen paper configurations

- `xor/runners/`: argv presets for the thin XOR entry points.
- `xor/search/`: bounded Sinkhorn and positive-scale fallback grids.
- `training_stage/default.yaml`: CIFAR-10 VGG endpoint protocol.
- `fashion_mnist/default.yaml`: 10-layer, width-512 Fashion-MNIST endpoint protocol.
- `dense_linear_stage/final_vgg{11,13,16,19}.yaml`: final CIFAR-10 alignment benchmarks.
- `dense_linear_stage/final_fashion_mnist.yaml`: final Fashion-MNIST alignment benchmark.
- `dense_linear_stage/_base/`: shared final-endpoint settings inherited by the five public configs; these are not standalone experiment entry points.

Hydra composition is tested for every public final config. The historical
12-by-12 training-stage configurations are archived and cannot be selected
from the active config tree. See [../../REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).
