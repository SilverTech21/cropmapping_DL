import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, cohen_kappa_score
import torch.nn.functional as F
from torch.autograd import Variable

from src.dataset import PASTIS_Dataset
from src import utils
from STCLN import UTAE, UTAEClassification
from logger import Logger, Printer, VisdomLogger
import argparse
from torch.utils.tensorboard import SummaryWriter

deviceID = [0]

# Set random seeds for reproducibility
torch.manual_seed(3407)
torch.backends.cudnn.deterministic = True


def collate_fn_wrapper(batch):
    """Module-level collate function to avoid pickling issues with lambda on Windows"""
    return utils.pad_collate(batch, pad_value=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--datadir", default="../../PASTIS_DATA", type=str, help="path to PASTIS_DATA directory")
    parser.add_argument('-b', "--batchsize", default=2, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=2, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=50, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument('-p', "--pretrain_pth", default="./checkpoints/pretrain/checkpoint_009.sttrans.tar", type=str, help="pretrain model path")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="path to snapshot to resume training")
    parser.add_argument('-c', "--checkpoint_dir", default="./checkpoints/finetune", type=str, help="directory to save checkpoints")
    return parser.parse_args()


def main(datadir, batchsize=2, workers=2, epochs=50, lr=1e-4, pretrain_pth=None, snapshot=None, checkpoint_dir=None):
    """
    Finetuning script for STCLN on PASTIS dataset
    
    Sampling strategy:
    - Train: 150 samples from fold [5]
    - Validation: 150 samples from fold [5]
    - Test: 7700 samples (not used in this script)
    """
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"Loading PASTIS dataset from: {datadir}")
    print("Finetuning on crop classification task")
    
    # Verify datadir exists
    if not os.path.exists(datadir):
        print(f"ERROR: Dataset directory not found: {datadir}")
        print("Please download PASTIS dataset and place it in the correct location")
        return
    
    # Load dataset for finetuning (fold 5)
    full_dataset = PASTIS_Dataset(
        folder=datadir,
        norm=True,
        folds=[5],  # Use fold 5 for finetuning
        target='semantic'
    )
    
    print(f"Loaded {len(full_dataset)} samples for finetuning")
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = max(1, int(0.5 * dataset_size))  # ~50% for train, 50% for val
    val_size = dataset_size - train_size
    
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, dataset_size)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Use num_workers=0 on Windows to avoid multiprocessing pickle issues
    traindataloader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        pin_memory=False,  # Disable pin_memory on CPU
        collate_fn=collate_fn_wrapper
    )
    
    valdataloader = DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        pin_memory=False,  # Disable pin_memory on CPU
        collate_fn=collate_fn_wrapper
    )
    
    print(f"Created DataLoaders with batch size {batchsize}")
    
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
    network = UTAEClassification(
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
        )
    )
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    start_epoch = 0
    
    # Load pretrained model
    if pretrain_pth is not None and os.path.exists(pretrain_pth):
        print(f"Loading pretrained model from {pretrain_pth}")
        try:
            network.utae.load_state_dict(torch.load(pretrain_pth, map_location='cpu'))
            print("Pretrained UTAE weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights ({e})")
    else:
        print("No pretrained weights provided, training from scratch")
    
    if snapshot is not None and os.path.exists(snapshot):
        print(f"Loading checkpoint from {snapshot}")
        checkpoint = torch.load(snapshot, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)
        network.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Move to GPU
    if torch.cuda.is_available():
        network = nn.DataParallel(network, device_ids=deviceID).cuda()
        loss_fn = loss_fn.cuda()
        print(f"Model moved to GPU {deviceID}")
    else:
        print("No GPU available, using CPU")
    
    # Setup tensorboard
    os.makedirs('./runs/finetune', exist_ok=True)
    writer = SummaryWriter('./runs/finetune')
    scaler = GradScaler()
    
    print("\nStarting finetuning...")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        logger.update_epoch(epoch)
        print(f"\nEpoch {epoch}/{epochs-1}")
        
        train_loss = train_epoch(traindataloader, network, optimizer, loss_fn, logger, vizlogger, scaler)
        val_loss = val_epoch(valdataloader, network, loss_fn, logger, vizlogger)
        
        writer.add_scalar("Training/Loss", train_loss, global_step=epoch)
        writer.add_scalar("Validation/Loss", val_loss, global_step=epoch)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if vizlogger is not None:
            try:
                data = logger.get_data()
                vizlogger.update(data)
            except:
                pass
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, network, optimizer, checkpoint_dir, 'best')
        
        # Save checkpoint every 10 epochs or at end
        if (epoch % 10 == 0 or epoch + 1 == epochs):
            save_checkpoint(epoch, network, optimizer, checkpoint_dir)
    
    print("\nFinetuning completed!")
    writer.close()


def train_epoch(dataloader, network, optimizer, loss_fn, logger, vizlogger, scaler):
    """Train for one epoch"""
    
    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    network.train()
    
    total_loss = 0
    num_batches = 0
    
    for iteration, data in enumerate(dataloader):
        try:
            (input_data, dates), labels = data
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
                dates = dates.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = network(input_data, dates)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            
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


def val_epoch(dataloader, network, loss_fn, logger, vizlogger):
    """Validate for one epoch"""
    
    logger.set_mode("test")
    network.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):
            try:
                (input_data, dates), labels = data
                
                if torch.cuda.is_available():
                    input_data = input_data.cuda()
                    dates = dates.cuda()
                    labels = labels.cuda()
                
                # Forward pass
                outputs, _ = network(input_data, dates)
                
                # Compute loss
                loss = loss_fn(outputs, labels)
                
                total_loss += loss.detach().cpu().item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in validation batch {iteration}: {e}")
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(epoch, model, optimizer, path, suffix=''):
    """Save model checkpoint"""
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if suffix:
        suffix = f"_{suffix}"
    
    # Save full checkpoint
    checkpoint_path = os.path.join(path, f"checkpoint_{epoch:03d}{suffix}.tar")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    
    main(
        datadir=args.datadir,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        pretrain_pth=args.pretrain_pth,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir
    )
