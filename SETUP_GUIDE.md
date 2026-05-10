# STCLN Project Setup Guide

## Directory Structure

```
DL project/
├── STCLN/
│   ├── PASTIS/
│   │   ├── src/                          # NEW: Dataset and utils modules
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py               # PASTIS dataset loader
│   │   │   └── utils.py                 # Utility functions
│   │   ├── STCLN.py                     # Model architecture with Swin Transformer
│   │   ├── pretraining_STCLN.py         # FIXED: Pretraining script
│   │   ├── finetuning_STCLN.py          # Finetuning script
│   │   ├── test_STCLN.py                # Testing script
│   │   └── logger.py                    # Logging utilities
│   └── PASTIS_DATA/                     # Dataset directory (download from Zenodo)
│       ├── DATA_S2/                     # Sentinel-2 observations (.npy files)
│       ├── ANNOTATIONS/                 # Semantic labels (.npy files)
│       ├── INSTANCE_ANNOTATIONS/        # Instance labels
│       ├── metadata.geojson             # Metadata with fold and date info
│       └── NORM_S2_patch.json          # Normalization parameters
```

## Dataset Sampling Strategy

Based on the research paper, the dataset is split as follows:

- **Pretraining**: ~7000 samples from folds [1,2,3,4]
  - Task: Self-supervised masked spectral band prediction
  - Goal: Learn spatiotemporal patterns without labels

- **Finetuning**:
  - Train: 150 samples from fold [5]
  - Validation: 150 samples (subset of fold [5])
  - Test: 7700 samples

## Changes Made

### 1. Created `src/dataset.py`
- Implements `PASTIS_Dataset` class
- Handles loading Sentinel-2 observations (11 bands)
- Loads semantic labels
- Supports fold-based sampling
- Handles max_samples for limiting dataset size

### 2. Created `src/utils.py`
- `pad_collate()`: Handles variable-length time series
- `get_train_val_test_splits()`: Dataset splitting utility
- `normalize_data()`: Normalization helper
- `apply_ndvi()`: NDVI calculation

### 3. Updated `pretraining_STCLN.py`
- Fixed imports (removed sys.path.append)
- Proper dataset loading with sampling
- Added error handling
- Improved logging and checkpointing
- 7000 sample limit for pretraining

### 4. Updated `STCLN.py`
- Integrated Swin Transformer for spatial encoding
- Fixed dimension mismatches
- Added spectral attention mechanism
- Proper temporal modeling with LTAE

## Running the Code

### 1. Download PASTIS Dataset

Download from: https://zenodo.org/record/5012942

Extract to: `DL project/STCLN/PASTIS_DATA/`

Expected structure:
```
PASTIS_DATA/
├── DATA_S2/
│   ├── 10000_S2.npy
│   ├── 10056_S2.npy
│   └── ... (thousands more)
├── ANNOTATIONS/
│   ├── 10000_agri_vec.npy
│   ├── 10056_agri_vec.npy
│   └── ... (thousands more)
└── metadata.geojson
```

### 2. Pretraining

```bash
cd DL project/STCLN/PASTIS
python pretraining_STCLN.py \
    --datadir ../../PASTIS_DATA \
    --batchsize 2 \
    --workers 2 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints/pretrain \
    --max_pretrain_samples 7000
```

### 3. Finetuning (After Pretraining)

```bash
python finetuning_STCLN.py \
    --datadir ../../PASTIS_DATA \
    --batchsize 2 \
    --workers 2 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --pretrain_pth ./checkpoints/pretrain/checkpoint_009.sttrans.tar \
    --checkpoint_dir ./checkpoints/finetune
```

## Model Architecture

**Spatial Encoder**: Swin Transformer (Tiny)
- Input: 11-band Sentinel-2 images (32x32)
- Output: 96-channel features at 8x8 resolution

**Temporal Encoder**: LTAE (Lightweight Temporal Attention Encoder)
- Processes temporal sequences with multi-head self-attention
- Learns temporal patterns across time steps

**Head Task**: 
- Pretraining: Masked band prediction (self-supervised)
- Finetuning: Crop type classification

**Spectral Attention**:
- Channel-wise attention for focusing on relevant bands

## Hardware Requirements

- **GPU**: 6GB VRAM (tested with batch size 2)
- **RAM**: 8GB+
- **Storage**: 50GB for PASTIS dataset

## Key Features

✅ Swin Transformer for spatial encoding
✅ Temporal Attention (LTAE) for temporal modeling
✅ Spectral Attention for band-wise focus
✅ Self-supervised pretraining with masked prediction
✅ Proper dataset sampling and splitting
✅ Error handling and validation
✅ Checkpoint saving and loading

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
- Ensure you're running from `DL project/STCLN/PASTIS/` directory
- Check that `src/` folder exists with `__init__.py`

### "FileNotFoundError: PASTIS_DATA not found"
- Download dataset from Zenodo
- Extract to correct location: `DL project/STCLN/PASTIS_DATA/`
- Verify metadata.geojson exists

### GPU Out of Memory
- Reduce batch size: `--batchsize 1`
- Reduce max_pretrain_samples: `--max_pretrain_samples 3000`
- Reduce workers: `--workers 1`

### Visdom Logger Errors
- Install visdom: `pip install visdom`
- Or ignore (logger will work without Visdom)

## Performance Notes

- **Pretraining**: ~1-2 hours for 10 epochs with 7000 samples (batch size 2)
- **Finetuning**: ~30 minutes for 50 epochs with 150 samples
- **Memory usage**: ~4-5 GB with batch size 2

## References

- PASTIS Dataset: https://github.com/VSainteuf/pastis-benchmark
- Paper: "Spatiotemporal masked pre-training for advancing crop mapping"
- Swin Transformer: https://github.com/microsoft/Swin-Transformer
