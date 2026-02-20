"""
Baseline dataset helper package.

Keep this module lightweight: do not import heavy optional dependencies (e.g.
albumentations) at package import time. Downstream configs should reference
concrete modules, e.g.:

- `baseline.dataset.dataset_code.semseg.SemanticSegmentationDataset`
- `baseline.dataset.dataset_code.utils.VoxelizeCollate`
"""

__all__: list[str] = []
