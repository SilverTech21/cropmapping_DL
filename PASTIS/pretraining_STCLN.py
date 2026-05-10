import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import PASTIS_Dataset
from src import utils
from STCLN import UTAE, UTAEPrediction
from logger import Logger, Printer, VisdomLogger
import argparse
from torch.utils.tensorboard import SummaryWriter

deviceIds = [0]


def collate_fn_wrapper(batch):
    """Module-level collate function to avoid pickling issues with lambda on Windows"""
    return utils.pad_collate(batch, pad_value=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--datadir", default="../../PASTIS_DATA", type=str, help="path to PASTIS_DATA directory")
    parser.add_argument('-b', "--batchsize", default=2, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=2, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=10, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default="./checkpoints/pretrain", type=str, help="directory to save checkpoints")
    parser.add_argument('--max_pretrain_samples', default=7000, type=int, help="max samples for pretraining")
    return parser.parse_args()


def main(datadir, batchsize=2, workers=2, epochs=10, lr=1e-4, snapshot=None, checkpoint_dir=None, max_pretrain_samples=7000):
    """
    Pretraining script for STCLN on PASTIS dataset
    
    Sampling strategy:
    - Pretrain: ~7000 samples from folds [1,2,3,4]
    - Uses self-supervised masked prediction task
    """
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"Loading PASTIS dataset from: {datadir}")
    print(f"Max pretrain samples: {max_pretrain_samples}")
    
    # Verify datadir exists
    if not os.path.exists(datadir):
        print(f"ERROR: Dataset directory not found: {datadir}")
        print("Please download PASTIS dataset and place it in the correct location")
        return
    
    # Load full dataset with fold sampling
    full_dataset = PASTIS_Dataset(
        folder=datadir,
        norm=True,
        folds=[1, 2, 3, 4],  # Use multiple folds for pretraining
        target='semantic',
        max_samples=max_pretrain_samples  # Sample ~7000
    )
    
    print(f"Loaded {len(full_dataset)} samples for pretraining")
    
    # Use num_workers=0 on Windows to avoid multiprocessing pickle issues
    traindataloader = DataLoader(
        full_dataset, 
        batch_size=batchsize, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        pin_memory=False,  # Disable pin_memory on CPU
        collate_fn=collate_fn_wrapper
    )
    
    print(f"Created DataLoader with batch size {batchsize}")
    
    # Setup logging
    logger = Logger(columns=["loss"], modes=["train", "test"])
    vizlogger = None
    try:
        vizlogger = VisdomLogger()
        print("Visdom logger initialized")
    except Exception as e:
        print(f"Warning: Visdom logger not available ({e})")
    
    # Model initialization
    print("Initializing model...")
    network = UTAEPrediction(
        UTAE(
            n_channels=10,  # PASTIS S2 uses 10 Sentinel-2 bands
            n_classes=20,
            bilinear=True,
            encoder_widths=[32, 256],
            decoder_widths=[32, 256],
            agg_mode="att_mean",
            n_head=8,
            d_model=256,
            d_k=32
        ),
        num_features=10,  # Predict 10 bands
        dropout=0.4
    )
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')
    
    start_epoch = 0
    
    if snapshot is not None and os.path.exists(snapshot):
        print(f"Loading checkpoint from {snapshot}")
        checkpoint = torch.load(snapshot, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)
        network.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Move to GPU if enough memory is available, otherwise use CPU
    use_cuda = False
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(deviceIds[0]).total_memory
        if total_mem >= 8 * 1024 ** 3:
            use_cuda = True
        else:
            print(f"GPU detected with {total_mem / 1024 ** 3:.1f}GB memory, using CPU to avoid OOM.")
    if use_cuda:
        network = nn.DataParallel(network, device_ids=deviceIds).cuda()
        loss_fn = loss_fn.cuda()
        print(f"Model moved to GPU {deviceIds}")
    else:
        print("Using CPU for training")
    
    # Setup tensorboard
    os.makedirs('./runs/pretrain', exist_ok=True)
    writer = SummaryWriter('./runs/pretrain')
    scaler = GradScaler()
    
    print("\nStarting pretraining...")
    print("="*60)
    
    for epoch in range(start_epoch, epochs):
        logger.update_epoch(epoch)
        print(f"\nEpoch {epoch}/{epochs-1}")
        
        epoch_loss = train_epoch(traindataloader, network, optimizer, loss_fn, logger, vizlogger, scaler)
        
        writer.add_scalar("Training/Loss", epoch_loss, global_step=epoch)
        print(f"Epoch {epoch} Loss: {epoch_loss:.6f}")
        
        if vizlogger is not None:
            try:
                data = logger.get_data()
                vizlogger.update(data)
            except:
                pass
        
        # Save checkpoint
        if (epoch % 10 == 0 or epoch + 1 == epochs):
            save_checkpoint(epoch, network, optimizer, checkpoint_dir)
    
    print("\nPretraining completed!")
    writer.close()


def train_epoch(dataloader, network, optimizer, loss_fn, logger, vizlogger, scaler):
    """Train for one epoch"""
    
    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    network.train()
    
    device = next(network.parameters()).device  # Get device from model
    total_loss = 0
    num_batches = 0
    
    for iteration, data in enumerate(dataloader):
        try:
            (input_data, dates), label = data
            
            # Move data to same device as model
            input_data = input_data.to(device)
            dates = dates.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            # Input shape: (B, T, C, H, W)
            # Create cluster IDs from NDVI (use B8 - B4 = NIR - RED)
            # Indices: B4=2 (RED), B8=6 (NIR) in Sentinel-2
            cluster_ids_x = (input_data[:, :, 6, :, :] - input_data[:, :, 2, :, :]) / (
                input_data[:, :, 2, :, :] + input_data[:, :, 6, :, :] + 1e-20
            )
            cluster_ids_x = cluster_ids_x.gt(0.2).int().unsqueeze(2).repeat(1, 1, input_data.shape[2], 1, 1)
            
            # Forward pass
            output, x2, target, mask = network.forward(input_data, cluster_ids_x, dates)
            
            # Downsample the original target and mask to the model output resolution
            if target.shape[-2:] != output.shape[-2:]:
                target = F.interpolate(
                    target.reshape(-1, target.shape[2], target.shape[3], target.shape[4]),
                    size=output.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).view(target.shape[0], target.shape[1], target.shape[2], output.shape[-2], output.shape[-1])
                mask = F.interpolate(
                    mask.float().reshape(-1, mask.shape[2], mask.shape[3], mask.shape[4]),
                    size=output.shape[-2:],
                    mode='nearest',
                ).view(mask.shape[0], mask.shape[1], mask.shape[2], output.shape[-2], output.shape[-1])
            
            # Compute loss only on masked regions
            loss = loss_fn(output, target.float())
            loss = (loss * (1 - mask).float()).sum() / ((1 - mask).sum() + 1e-20)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Logging
            stats = {"loss": loss.detach().cpu().item()}
            total_loss += stats["loss"]
            num_batches += 1
            
            if iteration % 10 == 0:
                printer.print(stats, iteration)
                logger.log(stats, iteration)
                if vizlogger is not None:
                    try:
                        vizlogger.plot_steps(logger.get_data())
                    except:
                        pass
        
        except Exception as e:
            print(f"Error in batch {iteration}: {e}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(epoch, model, optimizer, path):
    """Save model checkpoint"""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save full checkpoint
    checkpoint_path = os.path.join(path, f"checkpoint_{epoch:03d}.tar")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    # Save UTAE weights separately
    utae_path = os.path.join(path, f"checkpoint_{epoch:03d}.sttrans.tar")
    utae_module = model.module.utae if isinstance(model, nn.DataParallel) else model.utae
    torch.save(utae_module.state_dict(), utae_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    
    main(
        datadir=args.datadir,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir,
        max_pretrain_samples=args.max_pretrain_samples
    )
