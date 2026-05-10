import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch, pad_value=0):
    """
    Collate function for DataLoader that handles variable-length time series
    
    Args:
        batch: List of tuples ((s2_data, dates), label)
        pad_value: Value to use for padding
        
    Returns:
        (padded_s2, dates_list), labels: Padded tensors and labels
    """
    
    s2_data_list = []
    dates_list = []
    labels_list = []
    
    for (s2, dates), label in batch:
        # s2 shape: (T, C, H, W)
        # dates shape: (T,)
        
        # Ensure s2 is a tensor
        if not isinstance(s2, torch.Tensor):
            s2 = torch.from_numpy(s2).float()
        
        # Ensure dates is a tensor
        if not isinstance(dates, torch.Tensor):
            dates = torch.from_numpy(dates).long()
        else:
            dates = dates.long()
        
        s2_data_list.append(s2)
        dates_list.append(dates)
        labels_list.append(label)
    
    # Find max dimensions in the batch
    max_time = max([s2.shape[0] for s2 in s2_data_list])
    max_channels = max([s2.shape[1] for s2 in s2_data_list])
    max_height = max([s2.shape[2] for s2 in s2_data_list])
    max_width = max([s2.shape[3] for s2 in s2_data_list])
    
    # Pad each sample to the same (T, C, H, W) dimensions
    padded_s2_list = []
    for s2 in s2_data_list:
        if s2.dim() != 4:
            raise ValueError(f"Expected s2 tensor of shape (T, C, H, W), got {s2.shape}")
        t, c, h, w = s2.shape
        if (c, h, w) != (max_channels, max_height, max_width):
            pad_c = max_channels - c
            pad_h = max_height - h
            pad_w = max_width - w
            # Pad: (W, H, C, T) order for F.pad -> (left, right, top, bottom, front, back)
            pad = (0, pad_w, 0, pad_h, 0, pad_c)
            s2 = torch.nn.functional.pad(s2, pad, value=pad_value)
        if t < max_time:
            pad_size = max_time - t
            padding = torch.ones(pad_size, s2.shape[1], s2.shape[2], s2.shape[3], dtype=s2.dtype, device=s2.device) * pad_value
            s2 = torch.cat([s2, padding], dim=0)
        padded_s2_list.append(s2)
    
    # Stack into batch
    batch_s2 = torch.stack(padded_s2_list)  # (B, T_max, C, H, W)
    
    # Pad dates to match s2 time dimension
    padded_dates_list = []
    for dates in dates_list:
        # Ensure dates is 1D tensor
        if not isinstance(dates, torch.Tensor):
            dates = torch.from_numpy(dates).long()
        dates = dates.view(-1)
        
        if dates.shape[0] < max_time:
            pad_size = max_time - dates.shape[0]
            # Pad with 0 (no observation)
            padded_dates = torch.cat([
                dates,
                torch.zeros(pad_size, dtype=dates.dtype)
            ], dim=0)
        elif dates.shape[0] > max_time:
            # Truncate if longer than max_time
            padded_dates = dates[:max_time]
        else:
            padded_dates = dates
        padded_dates_list.append(padded_dates)
    
    # Stack dates
    batch_dates = torch.stack(padded_dates_list)  # (B, T_max)
    batch_labels = torch.stack(labels_list)  # (B, H, W)
    
    return (batch_s2, batch_dates), batch_labels


def get_train_val_test_splits(dataset_size, train_ratio=0.15, val_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        dataset_size: Total number of samples
        train_ratio: Fraction for training (default 0.15 -> 150 out of ~1000)
        val_ratio: Fraction for validation (default 0.15 -> 150 out of ~1000)
        
    Returns:
        train_indices, val_indices, test_indices
    """
    
    total_indices = np.arange(dataset_size)
    np.random.shuffle(total_indices)
    
    train_size = max(1, int(dataset_size * train_ratio))
    val_size = max(1, int(dataset_size * val_ratio))
    
    train_indices = total_indices[:train_size]
    val_indices = total_indices[train_size:train_size + val_size]
    test_indices = total_indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def normalize_data(data, mean=None, std=None):
    """
    Normalize data using provided mean and std
    
    Args:
        data: Input tensor (T, C, H, W) or (B, T, C, H, W)
        mean: Per-channel mean
        std: Per-channel std
        
    Returns:
        Normalized tensor
    """
    
    if mean is not None and std is not None:
        mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
        std = torch.tensor(std, dtype=data.dtype, device=data.device)
        
        # Reshape for broadcasting
        if data.dim() == 5:  # (B, T, C, H, W)
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
        elif data.dim() == 4:  # (T, C, H, W)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        
        return (data - mean) / (std + 1e-8)
    
    return data


def apply_ndvi(s2_data):
    """
    Calculate NDVI from Sentinel-2 data
    Sentinel-2 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, CLP (11 channels)
    NDVI = (NIR - RED) / (NIR + RED) where NIR=B8(index 6), RED=B4(index 2)
    
    Args:
        s2_data: (T, C, H, W) or (B, T, C, H, W)
        
    Returns:
        NDVI: (T, 1, H, W) or (B, T, 1, H, W)
    """
    
    if s2_data.dim() == 5:  # (B, T, C, H, W)
        nir = s2_data[:, :, 6, :, :]  # B8
        red = s2_data[:, :, 2, :, :]  # B4
    else:  # (T, C, H, W)
        nir = s2_data[:, 6, :, :]
        red = s2_data[:, 2, :, :]
    
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    if s2_data.dim() == 5:
        ndvi = ndvi.unsqueeze(2)  # (B, T, 1, H, W)
    else:
        ndvi = ndvi.unsqueeze(1)  # (T, 1, H, W)
    
    return ndvi
