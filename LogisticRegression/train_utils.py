import numpy as np
from typing import Tuple

from LogisticRegression.model import LinearModel
from utils import get_data_batch


def calculate_loss(
        model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
    
    Returns:
        loss: float, loss of the model
    '''
    y_preds = model(X).squeeze()
    # Clip predictions to avoid log(0)
    y_preds = np.clip(y_preds, 1e-10, 1 - 1e-10)
    if is_binary:
        loss = -np.mean(y * np.log(y_preds) + (1 - y) * np.log(1 - y_preds)) # binary cross-entropy loss
    else:
        #raise NotImplementedError('Calculate cross-entropy loss here')
        # Assuming y is one-hot encoded
        m = y.shape[0]


        # Obtain the predictions by passing X through the model
        p = model(X)  # The predicted probabilities
        # Clipping predictions to avoid log(0)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        
        if y.ndim == 1:
            # If y is not one-hot encoded, you need to one-hot encode it
            y_one_hot = np.eye(model.out_dim)[y.astype(int)]  # model.out_dim should be the number of classes
        else:
            
            y_one_hot = y
        
        # Cross-entropy loss
        log_likelihood = -np.log(p[np.arange(m), y_one_hot.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
    return loss


def calculate_accuracy(
        model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    y_preds = model(X).squeeze()
    if is_binary:
        acc = np.mean((y_preds > 0.5) == y) # binary classification accuracy
    else:
        #raise NotImplementedError('Calculate accuracy for multi-class classification here')
        predicted_labels = np.argmax(model(X), axis=1)
        actual_labels = np.argmax(y, axis=1) if y.ndim > 1 else y
        acc = np.mean(predicted_labels == actual_labels)
    return acc


def evaluate_model(
        model: LinearModel, X: np.ndarray, y: np.ndarray,
        batch_size: int, is_binary: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data and return the loss and accuracy.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
        batch_size: int, batch size for evaluation
    
    Returns:
        loss: float, loss of the model
        acc: float, accuracy of the model
    '''
    ## get predicitions
    #raise NotImplementedError(
        #'Get predictions in batches here (otherwise memory error for large datasets)')
    
    ## calculate loss and accuracy
    #loss = calculate_loss(model, X, y, is_binary)
    #acc = calculate_accuracy(model, X, y, is_binary)

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        batch_loss = calculate_loss(model, X_batch, y_batch, is_binary)
        batch_acc = calculate_accuracy(model, X_batch, y_batch, is_binary)

        total_loss += batch_loss * X_batch.shape[0]
        total_acc += batch_acc * X_batch.shape[0]
        total_samples += X_batch.shape[0]

    average_loss = total_loss / total_samples
    average_acc = total_acc / total_samples

    return average_loss, average_acc
    
    #return loss, acc


def fit_model(
        model: LinearModel, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, num_iters: int,
        lr: float, batch_size: int, l2_lambda: float,
        grad_norm_clip: float, is_binary: bool = False,
        decay_factor=0.6,  # New parameter for learning rate decay
        decay_patience=20,  # New parameter for patience before decay
) -> Tuple[list, list, list, list]:
    '''
    Fit the model on the given training data and return the training and validation
    losses and accuracies.

    Args:
        model: LinearModel, the model to be trained
        X_train: np.ndarray, features for training
        y_train: np.ndarray, labels for training
        X_val: np.ndarray, features for validation
        y_val: np.ndarray, labels for validation
        num_iters: int, number of iterations for training
        lr: float, learning rate for training
        batch_size: int, batch size for training
        l2_lambda: float, L2 regularization for training
        grad_norm_clip: float, clip value for gradient norm
        is_binary: bool, if True, use binary classification
    
    Returns:
        train_losses: list, training losses
        train_accs: list, training accuracies
        val_losses: list, validation losses
        val_accs: list, validation accuracies
    '''
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    learning_rate = lr  # Starting learning rate
    no_improve_epochs = 0  # Counter for epochs without improvement    

    #early stopping with patience part which i implemented
    best_val_loss = np.inf
    patience = 100  
    patience_counter = 0
    for i in range(num_iters + 1):
        # get batch
        X_batch, y_batch = get_data_batch(X_train, y_train, batch_size)
        
        # get predicitions
        y_preds = model(X_batch)
        
        # calculate loss
        loss = calculate_loss(model, X_batch, y_batch, is_binary)
        
        # calculate accuracy
        acc = calculate_accuracy(model, X_batch, y_batch, is_binary)
        
        # calculate gradient
        if is_binary:
            # Ensure y_batch is a column vector
            if y_batch.ndim == 1:
                y_batch = y_batch[:, np.newaxis]
            grad_W = X_batch.T @ (y_preds - y_batch) / batch_size
            grad_b = np.mean(y_preds - y_batch)
            # Make sure that grad_W is reshaped to match the shape of model.W
            grad_W = grad_W.reshape(model.W.shape)
        else:
            #ToDo
            # Assuming y is one-hot encoded
            # Softmax classification logic
            probs = np.clip(model(X_batch), 1e-10, 1 - 1e-10)

            # Calculate the gradient for softmax regression
            # Ensure y_batch is a 2D one-hot encoded matrix
            if y_batch.ndim == 1:
                y_batch = np.eye(model.out_dim)[y_batch]  # One-hot encode y_batch
            # Gradient of cross-entropy loss w.r.t. weights
            grad_W = X_batch.T @ (probs - y_batch) / batch_size
            #ToDo
            # Gradient of cross-entropy loss w.r.t. biases
            grad_b = np.mean(probs - y_batch, axis=0)        
        
        # regularization
        grad_W += l2_lambda * model.W
        grad_b += l2_lambda * model.b
        
        ## clip gradient norm
        #raise NotImplementedError('Clip gradient norm here')

        grad_norm = np.linalg.norm(grad_W)
        if grad_norm > grad_norm_clip:
            grad_W = grad_W * grad_norm_clip / grad_norm
            grad_b = grad_b * grad_norm_clip / grad_norm

        
        ## update model
        #raise NotImplementedError('Update model here (perform SGD)')
            
        model.W -= learning_rate * grad_W
        model.b -= learning_rate * grad_b


        if i % 10 == 0:
            # append loss
            train_losses.append(loss)
            # append accuracy
            train_accs.append(acc)

            # evaluate model
            val_loss, val_acc = evaluate_model(
                model, X_val, y_val, batch_size, is_binary)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(
                f'Iter {i}/{num_iters} - Train Loss: {loss:.4f} - Train Acc: {acc:.4f}'
                f' - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}'
            )
            
            ## TODO: early stopping here if required
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # reset counter if improvement
            else:
                patience_counter += 1  # increment counter if no improvement

            # Learning rate decay logic
            if patience_counter >= decay_patience:
                learning_rate *= decay_factor
                print(f"Decayed learning rate to: {learning_rate}")
                patience_counter = 0  # Reset the patience counter after decay


            # Check if early stopping is triggered
            if patience_counter >= patience:
                print(f"Early stopping triggered at iteration {i} (patience={patience})")
                break  # exit from the training loop
    
    return train_losses, train_accs, val_losses, val_accs
