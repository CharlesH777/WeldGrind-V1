# WeldGrind V1

Research code for predicting surface-contact grinding trajectories from observed RGB-D point clouds and SAM regions.

The source includes point-cloud geometry, surface-seed selection, trajectory DiT models, process-conditioned direct prediction with an ROI lattice, observed height-map model variants, and local surface filtering. The direct model predicts a complete 1024-point path and phase labels from fixed zero queries at t=0. The model variants include the base network, a deeper DiT, and a deeper selector.

## Install

```bash
python -m pip install -e ".[test,filter]"
```

Python 3.10 or later is required. PyTorch runs on CPU or a supported CUDA device. SAM base weights and action-model checkpoints are separate dependencies and are not included in this source release.

## Dataset

[WeldGrind-Dataset](https://huggingface.co/datasets/charlesH777/WeldGrind-Dataset) contains 2800 synthetic rendered states from 190 base workpieces, including original RGB-D, calibration, SAM predictions, observed point clouds and contact-path annotations. Download and extract `weldgrind_dataset_20261110_public_v1.zip` from that dataset repository.

The split contains 2260 training samples (1000 previously seen replay samples and 1260 new samples), 360 validation samples, and 180 test samples. The new geometry bases are disjoint across splits. Historical validation/test labels in the replay cohort do not define an independent holdout.

```python
from pathlib import Path
from weldgrind_v1.program_data import ProgramDataset

root = Path("weldgrind_dataset_20261110_public_v1")
data = ProgramDataset(root / "program_inputs/train.jsonl", seed=20261109,
                      max_points=2048, trajectory_points=1024)
sample = data[0]
print(len(data), sample["sample_id"], sample["trajectory"].shape)
```

World coordinates use millimetres. `ProgramDataset` returns normalized training tensors together with the observed centre and scale. Ground-truth trajectories and phase labels are supervision; they are not inference conditions. The saved XYZ paths represent surface contact, with execution infeed handled separately.

## Models and inference

`program_model.py` implements the surface-conditioned trajectory model. `roi_lattice_direct_program_model.py` provides the current direct model and its `make_roi_lattice_direct_program_model` factory; `heightmap_models.py` implements field and diffusion variants. `physical_trajectory_filter.py` and `background_surface.py` constrain predicted paths using observed surfaces.

The strict ROI-lattice checkpoint loader validates its objective identity and bound process/canonical-supervision manifests. Those additional training caches are not part of the original-GT dataset archive, so reproducing that training continuation requires the corresponding caches and checkpoint. The model factory and original-GT data loader can be used independently.

This is research code. Current held-out trajectory accuracy is still under evaluation; this release does not establish validated machining performance.

## Checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

See `SOURCE_MANIFEST.json` for the SHA256 inventory of the released model modules.
