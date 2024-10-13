import torch
from argparse import Namespace
from typing import Union, Tuple, List


import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
from LogisticRegression.train_utils import fit_model as fit_linear_model,\
    calculate_loss as calculate_linear_loss,\
    calculate_accuracy as calculate_linear_accuracy
import numpy as np

def calculate_loss(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        loss: float, loss of the model
    '''
    # raise NotImplementedError('Calculate negative-log-likelihood loss here')
    loss = torch.nn.CrossEntropyLoss()(y_logits, y.long()).item()   # the target tensor should be of the datatype long=> long int(64bit)
    return loss


def calculate_accuracy(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    # raise NotImplementedError('Calculate accuracy here')
    _, y_preds = torch.max(y_logits, 1)         
    acc = (y_preds == y).float().mean().item()
    return acc



def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        args: Namespace = None
) -> List[float]:
    '''
    Fit the contrastive model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - X: torch.Tensor, features
    - y: torch.Tensor, labels
    - num_iters: int, number of iterations for training
    - batch_size: int, batch size for training

    Returns:
    - losses: List[float], list of losses at each iteration
    '''
    patience = args.patience if args is not None else 10
    counter = 0
    best_loss = float('inf')
    # TODO: define the optimizer for the encoder only
    # using the Adam optimizer from torch.optim
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)


    # TODO: define the loss function
    # take the triplet loss from torch.
    triplet_loss= torch.nn.TripletMarginLoss(margin= 1.0, p=2)

    losses = []
    # iters=[]
    X= ptu.to_numpy(X)
    y= ptu.to_numpy(y)
    for i in range(num_iters):
        # get the batch of data
        # convert the tensors to numpy arrays
        
        X_a, X_p, X_n = next(get_contrastive_data_batch(X, y, batch_size))
        # Convert the numpy arrays to torch tensors
        X_a = ptu.from_numpy(X_a).float()
        X_p = ptu.from_numpy(X_p).float()
        X_n = ptu.from_numpy(X_n).float()
        # zero the gradients
        optimizer.zero_grad()

        # get the embeddings
        z_a = encoder(X_a)
        z_p = encoder(X_p)
        z_n = encoder(X_n)

        # calculate the loss
        loss = triplet_loss(z_a, z_p, z_n)

        # backpropagate the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # append the loss
        losses.append(loss.item())
        # iters.append(i)

        if i % 10 == 0:
            print(f'Iter: {i}, Loss: {loss.item()}', flush=True)
            # sys.stdout.flush()
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {i}')
                break

    return losses


def evaluate_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
        is_linear: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X: torch.Tensor, images
    - y: torch.Tensor, labels
    - batch_size: int, batch size for evaluation
    - is_linear: bool, whether the classifier is linear

    Returns:
    - loss: float, loss of the model
    - acc: float, accuracy of the model
    '''
    
    z= encoder(X)
    if is_linear:
        z= ptu.to_numpy(z)
    y_preds= classifier(z)
    if is_linear:
        return calculate_linear_loss(y_preds, y), calculate_linear_accuracy(y_preds, y)
   
    return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)


def fit_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        args: Namespace
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    '''
    Fit the model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X_train: torch.Tensor, training images
    - y_train: torch.Tensor, training labels
    - X_val: torch.Tensor, validation images
    - y_val: torch.Tensor, validation labels
    - args: Namespace, arguments for training

    Returns:
    - train_losses: List[float], list of training losses
    - train_accs: List[float], list of training accuracies
    - val_losses: List[float], list of validation losses
    - val_accs: List[float], list of validation accuracies
    '''
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = float('-inf')
    patience = args.patience
    counter = 0
    
    # Initialize variables for early stopping
    if args.mode == 'fine_tune_linear':
        z_train = np.zeros((0, args.z_dim))
        z_val = np.zeros((0, args.z_dim))
        for i in range(0, len(X_train), args.batch_size):
            X_batch = ptu.from_numpy(X_train[i:i+args.batch_size])
            z = ptu.to_numpy(encoder(X_batch))
            z_train=np.append(z_train, z, axis=0)
        
        for i in range(0, len(X_val), args.batch_size):
            end_idx = min(i + args.batch_size, len(X_val))
            X_batch = ptu.from_numpy(X_val[i:end_idx])
            z = ptu.to_numpy(encoder(X_batch))
            z_val=np.append(z_val, z, axis=0)
            
        train_losses, train_accs, val_losses, val_accs = fit_linear_model(
            classifier, z_train, y_train, z_val, y_val, args.num_iters, args.lr, args.batch_size, args.l2_lambda, args.grad_norm_clip)
        
    else:
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for i in range(args.num_iters):
            encoder.train()
            classifier.train()
            X_batch, y_batch = get_data_batch(X_train, y_train, args.batch_size)
            X_batch = ptu.from_numpy(X_batch).float()
            y_batch = ptu.from_numpy(y_batch).float()

            z = encoder(X_batch)
            classifier.to(z.device)
            y_preds = classifier(z)
            y_batch = y_batch.to(z.device).long()

            optimizer.zero_grad()
            loss = loss_fn(y_preds, y_batch)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                train_losses.append(loss.item())
                train_accs.append(calculate_accuracy(y_preds, y_batch))
                X_val = ptu.from_numpy(X_val).float()
                y_val = ptu.from_numpy(y_val).float()
                val_loss, val_acc = evaluate_model(encoder, classifier, X_val, y_val, args.batch_size, is_linear=False)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                print(f'Iter: {i}, Train Loss: {loss}, Train Acc: {train_accs[-1]}, Val Loss: {val_loss}, Val Acc: {val_acc}', flush=True)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping at epoch {i}')
                        break

    return train_losses, train_accs, val_losses, val_accs
