# Third-Party Licenses

This repository vendors third-party source code under `external/`. The vendored
license files are retained in place and apply to the corresponding directories.
Local changes are summarized in each directory's
`MODE_CONNECTIVITY_VENDOR_CHANGES.md` file.

| Path | Original source | Upstream base commit | License | Notes |
| --- | --- | --- | --- | --- |
| `external/dnn-mode-connectivity` | `https://github.com/timgaripov/dnn-mode-connectivity` | `f0bf253` | BSD 2-Clause | See `external/dnn-mode-connectivity/LICENSE`. Local changes are documented in `external/dnn-mode-connectivity/MODE_CONNECTIVITY_VENDOR_CHANGES.md`. |
| `external/pytorch-vgg-cifar10` | `https://github.com/chengyangfu/pytorch-vgg-cifar10` | `9c5da95` | MIT | See `external/pytorch-vgg-cifar10/LICENSE`. Local changes are documented in `external/pytorch-vgg-cifar10/MODE_CONNECTIVITY_VENDOR_CHANGES.md`. Generated `.pth` checkpoint files are ignored and are not part of the tracked vendored source. |
| `external/sinkhorn-rebasin` | `https://github.com/fagp/sinkhorn-rebasin` | `1c601d4` | MIT-style license text | See `external/sinkhorn-rebasin/LICENSE`. Local changes are documented in `external/sinkhorn-rebasin/MODE_CONNECTIVITY_VENDOR_CHANGES.md`. The license file is retained verbatim. |

These dependencies are tracked as ordinary source directories.
