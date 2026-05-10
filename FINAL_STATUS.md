# STCLN Project - Final Status Report

## ✅ Completed Tasks

### 1. Model Architecture
- **Swin Transformer Integration**: Replaced spatial CNN encoder with Swin Transformer (tiny variant)
  - Base model: `timm.create_model('swin_tiny_patch4_window7_224')`
  - Dimensions: d_model=256, encoder_widths=[32, 256]
  
- **Spectral Attention Module**: Added custom attention mechanism
  - Operates on spectral (channel) dimension
  - Integrated into UTAE architecture
  
- **Temporal Components**: Maintained LTAE for temporal feature aggregation

### 2. Code Quality & Syntax
✓ **finetuning_STCLN.py** - Syntax valid
- Proper imports from src.dataset and STCLN
- Main function with dataset sampling (150 train/val from fold 5)
- Train/validation loops with GPU support
- Checkpoint saving functionality

✓ **pretraining_STCLN.py** - Syntax valid
- Self-supervised masked prediction task
- Dataset sampling: 7000 samples from folds [1,2,3,4]
- Efficient batch processing with 4x4 patch splitting
- GPU memory optimization

✓ **STCLN.py** - Syntax valid
- Core model definitions (UTAE, UTAEPrediction, UTAEClassification)
- Swin encoder with spectral attention
- Proper tensor dimension handling

### 3. Dataset Integration
- **src/dataset.py**: Custom PASTIS_Dataset loader
  - Fold-based sampling strategy
  - Normalization from metadata
  - Support for max_samples parameter
  
- **src/utils.py**: Data collation utilities
  - Pad_collate function for variable-length sequences
  - Proper tensor padding for batch processing

### 4. GPU Optimization
- Batch size: 2 (optimized for 6GB GPU)
- Single GPU training (device 0)
- AMP (Automatic Mixed Precision) support with GradScaler
- Reduced number of workers (2) to minimize memory overhead

### 5. Training Infrastructure
- **Logger**: Training metrics logging with numpy support
- **TensorBoard**: Integration for loss visualization
- **Checkpoint Management**: Epoch-based model saving with best model tracking

## 📊 Sampling Strategy
```
Pretraining:  7000 samples from folds [1, 2, 3, 4]
Training:       150 samples from fold [5]
Validation:     150 samples from fold [5]
Testing:       7700 samples (fold [6] - not used in finetuning)
```

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install timm tensorboard visdom
pip install scikit-learn rasterio
```

### Pretraining
```bash
python pretraining_STCLN.py \
  -data /path/to/PASTIS_DATA \
  -b 2 \
  -w 2 \
  -e 100 \
  -l 1e-4
```

### Finetuning
```bash
python finetuning_STCLN.py \
  -data /path/to/PASTIS_DATA \
  -b 2 \
  -w 2 \
  -e 50 \
  -l 1e-4 \
  -p ./checkpoints/pretrain/checkpoint_009.sttrans.tar
```

## 📁 Project Structure
```
STCLN/
├── PASTIS/
│   ├── STCLN.py                 # Core model architecture
│   ├── pretraining_STCLN.py     # Self-supervised pretraining script
│   ├── finetuning_STCLN.py      # Supervised fine-tuning script
│   ├── test_STCLN.py            # Testing/inference script
│   ├── logger.py                # Logging utilities
│   └── src/
│       ├── __init__.py
│       ├── dataset.py           # PASTIS dataset loader
│       └── utils.py             # Utility functions
├── PASTIS_DATA/                 # Dataset directory (download required)
└── checkpoints/                 # Model checkpoint directory
    ├── pretrain/
    └── finetune/
```

## ⚙️ Key Parameters
- **Model**: UTAE with Swin Transformer encoder
- **Input Channels**: 11 (Sentinel-2 bands)
- **Output Classes**: 20 (crop types)
- **Patch Size**: 4 (Swin window size: 7x7)
- **Model Dimension**: 256
- **Number of Heads**: 8
- **Key Dimension**: 32

## 📝 Notes
- All scripts have valid Python syntax ✓
- GPU memory optimized for 6GB systems
- Training time estimated: 4-6 hours for pretraining, 1-2 hours for finetuning
- Requires PASTIS dataset download from: https://zenodo.org/record/5012942

## 🔧 Dependencies Installed
- torch, torchvision, torchaudio
- timm (Swin Transformer models)
- tensorboard (training visualization)
- visdom (optional visualization)
- scikit-learn (metrics)
- rasterio (geospatial data handling)

## ✨ Implementation Highlights
1. **Swin Transformer**: State-of-the-art vision transformer for spatial feature extraction
2. **Spectral Attention**: Channel-wise attention for crop-specific spectral patterns
3. **Temporal Attention**: LTAE for multi-temporal feature aggregation
4. **Memory Efficient**: Optimized for 6GB GPU with careful batch sizing
5. **Modular Design**: Clean separation of dataset, model, and training code
