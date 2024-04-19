import torch
from typing import List, Tuple, Union
from argparse import Namespace
import numpy as np

from ContrastiveRepresentation.pytorch_utils import from_numpy
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
from ContrastiveRepresentation.model import Encoder
import ContrastiveRepresentation.pytorch_utils as ptu
from LogisticRegression.train_utils import fit_model as fit_linear_model, calculate_loss as calculate_linear_loss, calculate_accuracy as calculate_linear_accuracy


def calculate_loss(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculate the loss using negative log likelihood loss (Cross-Entropy loss).
    """
    
    criterian = torch.nn.NLLLoss()
    loss = criterian(y_logits, y.long())
    return loss

def calculate_accuracy(y_logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculate the accuracy of the predictions.
    """

    y_pred = torch.nn.functional.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    acc = (y_pred == y).sum().item() / y.shape[0]
    return acc



def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3
) -> None:
    """
    Fit the contrastive model using TripletMarginLoss.
    """
    print("training started")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    loss_function = torch.nn.TripletMarginLoss()
    losses = []
    X = ptu.to_numpy(X)
    y = ptu.to_numpy(y)

    i=0

    for _ in range(num_iters):
        # Sample a batch of triplets
        X_anchor, X_positive, X_negative = get_contrastive_data_batch(X, y, batch_size)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        anchor_embeds = encoder(X_anchor)
        positive_embeds = encoder(X_positive)
        negative_embeds = encoder(X_negative)
        
        # Compute loss
        loss = loss_function(anchor_embeds, positive_embeds, negative_embeds)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        print(f"Iteration: {i}, loss: {loss}")
        i+=1
        if loss.item() < 0.015:
            break
        if i%10 == 0:
            torch.save(encoder.state_dict(), f"./models/encoder{i}.pth")
            print(f'Encoder saved to ./models/encoder{i}.pth')
    return losses



def evaluate_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
        is_linear: bool = False
) -> Tuple[float, float]:
    """
    Evaluate the model and return the loss and accuracy.
    """
    if is_linear:
        encoder.eval()
        z = encoder(X)
        y_preds = classifier(z)
        loss = calculate_linear_loss(y_preds, y)
        acc = calculate_linear_accuracy(y_preds, y)
        return loss, acc
    else:
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            y_preds = classifier(encoder(X))
    # HINT: use calculate_loss and calculate_accuracy functions for NN classifier and calculate_linear_loss and calculate_linear_accuracy functions for linear (softmax) classifier

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
    """
    Train the model and return the training and validation losses and accuracies.
    """
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    encoder.eval()
    print("fit model initialized")

    if args.mode == 'fine_tune_linear':
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        y_train = ptu.to_numpy(y_train)
        y_val = ptu.to_numpy(y_val)
        encoded_train = None
        encoded_val = None
        for i in range(0, len(X_train), args.batch_size):
            if i + args.batch_size > len(X_train):
                X_batch = X_train[i:].to(ptu.device)
            else:
                X_batch = X_train[i:i+args.batch_size].to(ptu.device)
            image_encoding = encoder(X_batch)
            image_encoding = ptu.to_numpy(image_encoding)
            if encoded_train is None:
                encoded_train = image_encoding
            else:
                encoded_train = np.concatenate((encoded_train, image_encoding))
        for i in range(0, len(X_val), args.batch_size):
            if i + args.batch_size > len(X_val):
                X_batch = X_val[i:].to(ptu.device)
            else:
                X_batch = X_val[i:i+args.batch_size].to(ptu.device)
            image_encoding = encoder(X_batch)
            image_encoding = ptu.to_numpy(image_encoding)
            if encoded_val is None:
                encoded_val = image_encoding
            else:
                encoded_val = np.concatenate((encoded_val, image_encoding))
        train_losses, train_accs, val_losses, val_accs = fit_linear_model(classifier, encoded_train, y_train, encoded_val, y_val, num_iters=args.num_iters, lr=args.lr, batch_size=args.batch_size, l2_lambda=args.l2_lambda, grad_norm_clip=args.grad_norm_clip, is_binary=args.mode == 'logistic')
        return train_losses, train_accs, val_losses, val_accs
    else:
        # TODO: define the optimizer
        # print(classifier)
        X_val = ptu.to_numpy(X_val)
        y_val = ptu.to_numpy(y_val)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
        num_batches = (X_train.shape[0] + args.batch_size - 1) // args.batch_size
        for i in range(args.num_iters):
            classifier.train()
            start = 0
            for j in range(num_batches):
                if j == num_batches-1:
                    pass
                else:
                    X_batch = X_train[start:start+args.batch_size]
                    y_batch = y_train[start:start+args.batch_size]
                
                image_encoding = encoder(X_batch)
                # print(len(image_encoding[0]))
                y_logits = classifier(image_encoding)
                loss = calculate_loss(y_logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_acc = calculate_accuracy(y_logits, y_batch)
                train_accs.append(train_acc)
                X_batch, y_batch = get_data_batch(X_val, y_val, args.batch_size)
                X_batch = ptu.from_numpy(X_batch)
                y_batch = ptu.from_numpy(y_batch)
                image_encoding = encoder(X_batch)
                y_logits = classifier(image_encoding)
                val_loss = calculate_loss(y_logits, y_batch)
                val_acc = calculate_accuracy(y_logits, y_batch)
                val_losses.append(val_loss.item())
                val_accs.append(val_acc)
            print(f"Epoch {i} Train Loss: {train_losses[-1]}, Train Acc: {train_accs[-1]}, Val Loss: {val_losses[-1]}, Val Acc: {val_accs[-1]}")
        print("MAX: ", max(val_accs), max(train_accs))
        return train_losses, train_accs, val_losses, val_accs
