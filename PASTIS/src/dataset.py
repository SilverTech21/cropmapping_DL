import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class PASTIS_Dataset(Dataset):
    """
    PASTIS Dataset for Satellite Time Series Classification
    Download from: https://zenodo.org/record/5012942
    
    Structure:
    PASTIS_DATA/
        DATA_S2/         (Sentinel-2 observations)
        ANNOTATIONS/     (Labels)
        metadata.geojson (Metadata with fold and date info)
        NORM_S2_patch.json (Normalization parameters)
    """
    
    def __init__(self, folder="./PASTIS_DATA", 
                 norm=True, 
                 target='semantic', 
                 folds=None,
                 reference_date="2019-09-01",
                 max_samples=None):
        """
        Args:
            folder: Path to PASTIS_DATA directory
            norm: Whether to apply normalization
            target: 'semantic' for classification or other targets
            folds: List of fold numbers to use (1-5)
            reference_date: Reference date for temporal encoding
            max_samples: Maximum number of samples to use (for sampling)
        """
        
        self.folder = folder
        self.norm = norm
        self.target = target
        self.folds = folds if folds is not None else [1, 2, 3, 4, 5]
        self.max_samples = max_samples
        
        # Load metadata
        metadata_path = os.path.join(folder, 'metadata.geojson')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Build patch list from metadata
        self.patch_ids = []
        self.dates = []
        
        for feature in metadata['features']:
            fold = feature['properties']['Fold']
            if fold in self.folds:
                patch_id = feature['properties']['ID_PATCH']
                dates_dict = feature['properties']['dates-S2']
                # Convert dates dict to sorted list
                dates = sorted([int(dates_dict[key]) for key in dates_dict.keys()])
                
                self.patch_ids.append(patch_id)
                self.dates.append(dates)
        
        # Apply sampling if max_samples specified
        if max_samples is not None and len(self.patch_ids) > max_samples:
            indices = np.random.choice(len(self.patch_ids), max_samples, replace=False)
            self.patch_ids = [self.patch_ids[i] for i in indices]
            self.dates = [self.dates[i] for i in indices]
        
        # Load normalization parameters
        self.norm_params = None
        if norm:
            norm_path = os.path.join(folder, 'NORM_S2_patch.json')
            if os.path.exists(norm_path):
                with open(norm_path, 'r') as f:
                    self.norm_params = json.load(f)
    
    def __len__(self):
        return len(self.patch_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            (input, dates): input shape (T, C, H, W), dates shape (T,)
            label: semantic label
        """
        patch_id = self.patch_ids[idx]
        dates = np.array(self.dates[idx])
        
        # Load time series data
        s2_data = self._load_sentinel2_data(patch_id)
        
        # Load label
        label = self._load_label(patch_id)
        
        # Normalize data format to (T, C, H, W)
        if s2_data.ndim == 4:
            if s2_data.shape[-1] in (10, 11) and s2_data.shape[1] not in (10, 11):
                s2_data = np.transpose(s2_data, (0, 3, 1, 2))
            elif s2_data.shape[1] in (10, 11) and s2_data.shape[-1] not in (10, 11):
                # Already in (T, C, H, W) format
                pass
            else:
                # Data layout is ambiguous; preserve shape and let downstream validation catch errors
                pass
        
        # Ensure dates match s2_data time dimension
        # Truncate or pad dates to match T
        t_dim = s2_data.shape[0]
        if len(dates) > t_dim:
            dates = dates[:t_dim]  # Truncate to match s2_data
        elif len(dates) < t_dim:
            # Pad with 0 (shouldn't happen with proper data)
            dates = np.pad(dates, (0, t_dim - len(dates)), mode='constant', constant_values=0)
        
        # Apply normalization
        if self.norm and self.norm_params is not None:
            s2_data = self._normalize(s2_data, patch_id)
        
        # Convert to tensors
        s2_tensor = torch.from_numpy(s2_data).float()
        label_tensor = torch.from_numpy(label).long()
        dates_tensor = torch.from_numpy(dates).long()
        
        return (s2_tensor, dates_tensor), label_tensor
    
    def _load_sentinel2_data(self, patch_id):
        """Load Sentinel-2 data for a patch"""
        patch_str = str(patch_id)
        s2_path = os.path.join(self.folder, 'DATA_S2', f'S2_{patch_str}.npy')
        
        if os.path.exists(s2_path):
            return np.load(s2_path)
        else:
            # Return dummy data if file doesn't exist (for development)
            return np.random.randn(30, 10, 128, 128).astype(np.float32)
    
    def _load_label(self, patch_id):
        """Load semantic label for a patch"""
        patch_str = str(patch_id)
        label_path = os.path.join(self.folder, 'ANNOTATIONS', f'ParcelIDs_{patch_str}.npy')
        if not os.path.exists(label_path):
            # Fallback to legacy naming patterns
            patch_str5 = str(patch_id).zfill(5)
            label_path = os.path.join(self.folder, 'ANNOTATIONS', f'{patch_str5}_agri_vec.npy')
        
        if os.path.exists(label_path):
            label = np.load(label_path)
            return label
        else:
            # Return dummy label if file doesn't exist
            return np.zeros((128, 128), dtype=np.int32)
    
    def _normalize(self, data, patch_id):
        """Normalize Sentinel-2 data using provided parameters"""
        if self.norm_params is None:
            return data
        
        # Get normalization params from JSON (or use defaults)
        # Sentinel-2 data is typically normalized by band statistics
        # For now, use simple z-score normalization
        mean = np.array(self.norm_params.get('mean', [0]*11))
        std = np.array(self.norm_params.get('std', [1]*11))
        
        # Normalize each band: (data - mean) / std
        for c in range(data.shape[1]):
            if c < len(mean) and c < len(std) and std[c] > 0:
                data[:, c, :, :] = (data[:, c, :, :] - mean[c]) / (std[c] + 1e-8)
        
        return data
