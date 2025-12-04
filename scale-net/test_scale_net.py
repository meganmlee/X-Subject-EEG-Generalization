"""
Test script for ChannelWiseSpectralCLDNN + SE Block
Uses best_ssvep_model.pth checkpoint
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Import model and dataset functions from model_cldnn_se.py
from model_cldnn_se import ChannelWiseSpectralCLDNN
from dataset import load_dataset, TASK_CONFIGS, create_dataloaders


# Note: Using model and dataset functions from model_cldnn_se.py and dataset.py


# ==================== Evaluation ====================

def evaluate(model, loader, device, is_binary=False):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        is_binary: Whether this is binary classification (n_classes=2)
    
    Returns:
        accuracy, predictions, labels
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Prediction: binary uses sigmoid threshold, multi-class uses argmax
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()  # (B, 1) → (B,)
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return 100. * correct / total, np.array(all_preds), np.array(all_labels)


# ==================== Confusion Matrix ====================

def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Print confusion matrix as text in terminal
    Shows counts and percentages for each class
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)
    class_labels = [chr(65 + i) for i in range(n_classes)]
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    print(f"\n{title}")
    
    # Calculate column width (need space for 3-digit numbers)
    col_width = 4  # Each cell is 4 characters wide
    
    # Total width: "True\\Pred" (9) + n_classes * col_width + " Total" (6)
    total_width = 9 + n_classes * col_width + 6
    print("=" * total_width)
    
    # Header: Predicted labels
    header = "True\\Pred"
    for label in class_labels:
        header += f"{label:>{col_width}s}"
    header += " Total"
    print(header)
    print("-" * total_width)
    
    # Each row: True label, predicted counts, total
    for i in range(n_classes):
        row = f"   {class_labels[i]:>3s}  "
        total_true = cm[i].sum()
        for j in range(n_classes):
            count = cm[i, j]
            row += f"{count:>{col_width}d}"
        row += f"{total_true:>6d}"
        print(row)
    
    print("-" * total_width)
    
    # Column totals
    col_totals = cm.sum(axis=0)
    row = "  Total  "
    for total in col_totals:
        row += f"{total:>{col_width}d}"
    row += f"{len(y_true):>6d}"
    print(row)
    print("=" * total_width)
    
    # Summary statistics
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = 100. * correct / total
    
    print(f"\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i in range(n_classes):
        if cm[i, i] > 0 or cm[i].sum() > 0:
            cls_total = cm[i].sum()
            cls_correct = cm[i, i]
            cls_acc = 100. * cls_correct / cls_total if cls_total > 0 else 0.0
            print(f"  Class {i:2d} ({class_labels[i]}): {cls_acc:5.1f}% ({cls_correct}/{cls_total})")


# ==================== Main Test ====================

def main():
    # Config
    checkpoint_path = './checkpoints/best_ssvep_model.pth'
    task = 'SSVEP'
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Testing ChannelWiseSpectralCLDNN + SE Block")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # ====== Load Checkpoint First to Get Config ======
    print(f"\nLoading checkpoint to get model configuration...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print("  Found config in checkpoint")
    else:
        saved_config = {}
        print("  No config in checkpoint, using defaults")
    
    # Get task-specific defaults
    task_config = TASK_CONFIGS.get(task, {})
    
    # Merge configs: checkpoint config > task defaults > hardcoded defaults
    config = {
        'data_dir': saved_config.get('data_dir', task_config.get('data_dir', '/ocean/projects/cis250213p/shared/ssvep')),
        'num_seen': saved_config.get('num_seen', task_config.get('num_seen', 33)),
        'seed': saved_config.get('seed', 44),
        'n_classes': saved_config.get('n_classes', task_config.get('num_classes', 26)),
        
        # Model parameters from checkpoint
        'cnn_filters': saved_config.get('cnn_filters', 16),
        'lstm_hidden': saved_config.get('lstm_hidden', 128),
        'pos_dim': saved_config.get('pos_dim', 16),
        'dropout': saved_config.get('dropout', 0.3),
        'cnn_dropout': saved_config.get('cnn_dropout', 0.2),
        'use_hidden_layer': saved_config.get('use_hidden_layer', False),
        'hidden_dim': saved_config.get('hidden_dim', 64),
        
        # STFT parameters from checkpoint or task defaults
        'stft_fs': saved_config.get('stft_fs', task_config.get('sampling_rate', 250)),
        'stft_nperseg': saved_config.get('stft_nperseg', task_config.get('stft_nperseg', 128)),
        'stft_noverlap': saved_config.get('stft_noverlap', task_config.get('stft_noverlap', 112)),
        'stft_nfft': saved_config.get('stft_nfft', task_config.get('stft_nfft', 512)),
    }
    
    print(f"\nModel Configuration:")
    print(f"  n_classes: {config['n_classes']}")
    print(f"  cnn_filters: {config['cnn_filters']}")
    print(f"  lstm_hidden: {config['lstm_hidden']}")
    print(f"  dropout: {config['dropout']}, cnn_dropout: {config['cnn_dropout']}")
    print(f"  use_hidden_layer: {config['use_hidden_layer']}, hidden_dim: {config['hidden_dim']}")
    
    # ====== Load Data Using dataset.py ======
    print(f"\nLoading {task} dataset...")
    datasets = load_dataset(
        task=task,
        data_dir=config['data_dir'],
        num_seen=config['num_seen'],
        seed=config['seed']
    )
    
    stft_config = {
        'fs': config['stft_fs'],
        'nperseg': config['stft_nperseg'],
        'noverlap': config['stft_noverlap'],
        'nfft': config['stft_nfft']
    }
    
    print(f"\nSTFT Parameters:")
    print(f"  Sampling Rate: {stft_config['fs']} Hz")
    print(f"  nperseg: {stft_config['nperseg']} samples ({stft_config['nperseg']/stft_config['fs']:.3f} sec)")
    print(f"  noverlap: {stft_config['noverlap']} samples ({100*stft_config['noverlap']/stft_config['nperseg']:.1f}% overlap)")
    print(f"  nfft: {stft_config['nfft']}")
    
    # ====== Create Data Loaders ======
    loaders = create_dataloaders(
        datasets,
        stft_config,
        batch_size=batch_size,
        num_workers=4,
        augment_train=False  # No augmentation for testing
    )
    
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions from a sample
    sample_x, _ = next(iter(val_loader))
    _, n_channels, freq_bins, time_bins = sample_x.shape
    print(f"\nSTFT shape: ({n_channels}, {freq_bins}, {time_bins})")
    print(f"Val: {len(loaders['val'].dataset)}, "
          f"Test1 (Seen): {len(test1_loader.dataset) if test1_loader else 0}, "
          f"Test2 (Unseen): {len(test2_loader.dataset) if test2_loader else 0}")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    is_binary = (n_classes == 2)
    
    model = ChannelWiseSpectralCLDNN(
        freq_bins=freq_bins,
        time_bins=time_bins,
        n_channels=n_channels,
        n_classes=n_classes,
        cnn_filters=config['cnn_filters'],
        lstm_hidden=config['lstm_hidden'],
        pos_dim=config['pos_dim'],
        dropout=config['dropout'],
        cnn_dropout=config['cnn_dropout'],
        use_hidden_layer=config['use_hidden_layer'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {n_params:,}")
    
    # ====== Load Checkpoint ======
    print(f"\nLoading checkpoint weights...")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        print(f"  Task: {checkpoint.get('task', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Checkpoint loaded successfully!")
    
    # ====== Evaluate ======
    print(f"\n{'='*70}")
    print("Evaluating...")
    print(f"{'='*70}")
    
    val_acc, val_preds, val_labels = evaluate(model, val_loader, device, is_binary=is_binary)
    if test1_loader:
        test1_acc, test1_preds, test1_labels = evaluate(model, test1_loader, device, is_binary=is_binary)
    else:
        test1_acc, test1_preds, test1_labels = None, None, None
    
    if test2_loader:
        test2_acc, test2_preds, test2_labels = evaluate(model, test2_loader, device, is_binary=is_binary)
    else:
        test2_acc, test2_preds, test2_labels = None, None, None
    
    # ====== Results ======
    print(f"\n{'='*70}")
    print(f"RESULTS - {task} (CLDNN + SE Block)")
    print(f"{'='*70}")
    print(f"Validation Acc:  {val_acc:.2f}%")
    if test1_acc is not None:
        print(f"Test1 (Seen):    {test1_acc:.2f}%")
    if test2_acc is not None:
        print(f"Test2 (Unseen):  {test2_acc:.2f}%")
    print(f"{'='*70}")
    
    # ====== Confusion Matrix ======
    if test1_preds is not None and test1_labels is not None:
        print_confusion_matrix(
            test1_labels, test1_preds, 
            title="Confusion Matrix - Test1 (Seen Subjects)"
        )
    
    results = {
        'val_acc': val_acc,
        'test1_acc': test1_acc,
        'test2_acc': test2_acc
    }
    
    return results


if __name__ == "__main__":
    results = main()

