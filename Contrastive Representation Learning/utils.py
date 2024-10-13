import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_data(
        data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
        is_binary: bool = False, grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load CIFAR-10 dataset from the given path and return the images and labels.
    If is_linear is True, the images are reshaped to 1D array.
    If grayscale is True, the images are converted to grayscale.

    Args:
    - data_path: string, path to the dataset
    - is_linear: bool, whether to reshape the images to 1D array
    - is_binary: bool, whether to convert the labels to binary
    - grayscale: bool, whether to convert the images to grayscale

    Returns:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    '''
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
    X = X.transpose(0, 3, 1, 2)         # (N, H, W, C) -> (N, C, H, W)
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        X = convert_to_grayscale(X)
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    
    # HINT: rescale the images for better (and more stable) learning and performance

    return X, y


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    '''
    Convert the given images to grayscale.

    Args:
    - X: np.ndarray, images in RGB format

    Returns:
    - X: np.ndarray, grayscale images
    '''
    return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the given dataset into training and test sets.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - test_ratio: float, ratio of the test set

    Returns:
    - X_train: np.ndarray, training images
    - y_train: np.ndarray, training labels
    - X_test: np.ndarray, test images
    - y_test: np.ndarray, test labels
    '''
    print(f"Splitting the dataset with test ratio: {test_ratio}"
          f" and total size: {len(X)}")
    print(X.shape, y.shape)
    assert test_ratio < 1 and test_ratio > 0
    # Split the dataset into training and test sets
    idxs = np.random.permutation(len(X))
    X = X[idxs]
    y = y[idxs]
    split = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train, y_train, X_test, y_test


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''
    idxs = np.random.choice(len(X), size=batch_size, replace=False)
    return X[idxs], y[idxs]


# TODO: Read up on generator functions online
def get_contrastive_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''
    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Yield batches of anchor, positive, and negative samples
    while True:
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            X_a, X_p, X_n = [], [], []
            for j in range(len(batch_X)):
                idx = j
                X_a.append(batch_X[idx])
                same_class_indices = np.where(batch_y == batch_y[idx])[0]
                diff_class_indices = np.where(batch_y != batch_y[idx])[0]
                X_p.append(batch_X[np.random.choice(same_class_indices)])
                X_n.append(batch_X[np.random.choice(diff_class_indices)])
            yield np.array(X_a), np.array(X_p), np.array(X_n)    

def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
    - train_losses: list, training losses
    - val_losses: list, validation losses
    - title: str, title of the plot
    '''
    # if isinstance(train_losses, torch.Tensor):
    #     train_losses = train_losses.cpu().detach().numpy()
    # if isinstance(val_losses, torch.Tensor):
    #     val_losses = val_losses.cpu().detach().numpy()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/loss-{title}.png')
    plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
    - train_accs: list, training accuracies
    - val_accs: list, validation accuracies
    - title: str, title of the plot
    '''
    # if isinstance(train_losses, torch.Tensor):
    #     train_losses = train_losses.cpu().detach().numpy()
    # if isinstance(val_losses, torch.Tensor):
    #     val_losses = val_losses.cpu().detach().numpy()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/acc-{title}.png')
    plt.close()


def plot_tsne(
       z: np.ndarray, y: np.ndarray, batch_size: int 
) -> None:
    '''
    Plot the 2D t-SNE of the given representation.

    Args:
    - z: torch.Tensor, representation
    - y: torch.Tensor, labels
    '''
    tsne = TSNE(n_components=2, random_state=42, n_jobs=2)
    plt.figure(figsize=(10, 10))
    z_embedded = tsne.fit_transform(z)
    
    for j in range(10):
        idxs = np.where(y == j)[0]
        plt.scatter(z_embedded[idxs, 0], z_embedded[idxs, 1], label=str(j))
    
    plt.title('t-SNE of the Representation')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(f'images/tsne-{batch_size}.png')
    print('Saved t-SNE plot')
    plt.close()
     
def plot_contrep_losses(losses: list, title: str) -> None:
    '''
    Plot the contrastive loss.

    Args:
    - losses: list, contrastive loss
    - title: str, title of the plot
    '''
    plt.plot(losses, label="Contrastive Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(f'images/contrastive-loss-{title}.png')
    print(f"Saved contrastive loss plot")
    plt.close()
