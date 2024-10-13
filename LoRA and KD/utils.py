import torch
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.mps.deterministic = True


def get_data_loader(data_path, batch_size, tokenizer, shuffle=True, max_len=20):
    """
    Get a data loader for the training data.
    """
    data = np.loadtxt(data_path, delimiter='\t', dtype=str)
    X, y = data[:, -1], data[:, 1]
    X = tokenizer.batch_encode_plus(
        X.tolist(), max_length=max_len, truncation=True, padding='max_length')
    X, mask = X['input_ids'], X['attention_mask']
    X = torch.tensor(np.array(X))
    mask = torch.tensor(np.array(mask))
    y = torch.tensor(np.array(y, dtype=int))
    data = torch.utils.data.TensorDataset(X, mask, y)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def plot_losses(train_losses: list,
            val_losses: list,
            args: Namespace):
    """
    Plot training and validation losses.
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{args.mode}-{args.gpt_variant}-Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.savefig(f'plots/{args.sr_no}-{args.mode}-{args.gpt_variant}-losses.png')
    plt.show()

def plot_metrics(train_metrics: list,
            val_metrics: list,
            args: Namespace):
    """
    Plot training and validation metrics.
    """
    plt.figure()
    plt.plot(train_metrics, label='Train Accuracy')
    plt.plot(val_metrics, label='Val Accuracy')
    plt.title(f'{args.mode}-{args.gpt_variant}-Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()    
    plt.savefig(f'plots/{args.sr_no}-{args.mode}-{args.gpt_variant}-metrics.png')
    plt.show()
