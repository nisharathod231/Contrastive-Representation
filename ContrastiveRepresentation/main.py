import torch
from argparse import Namespace
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Assuming ContrastiveRepresentation.pytorch_utils imports correctly
import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data, train_test_split, plot_tsne, plot_losses, plot_accuracies
from ContrastiveRepresentation.model import Encoder, Classifier
from ContrastiveRepresentation.train_utils import fit_contrastive_model, fit_model
from LogisticRegression.model import SoftmaxRegression  # Assuming this is your linear classifier

def main(args: Namespace):
    torch.manual_seed(args.sr_no)
    X, y = get_data(args.train_data_path, is_linear=False)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    
    # Convert to PyTorch tensors
    X_train, y_train, X_val, y_val = map(lambda x: ptu.from_numpy(x), (X_train, y_train, X_val, y_val))

    encoder = Encoder(args.z_dim).to(ptu.device)
    classifier = None  # Default to None

    if args.mode == 'fine_tune_linear':
        classifier = SoftmaxRegression(input_dim=args.z_dim, output_dim=10).to(ptu.device)
    elif args.mode == 'fine_tune_nn':
        classifier = Classifier(input_dim=args.z_dim, num_classes=10).to(ptu.device)
    
    if args.mode == 'cont_rep':
        # Assume ContrastiveDataset is a custom Dataset class you've implemented
        train_dataset = ContrastiveDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        fit_contrastive_model(encoder, train_loader, args)
        
        # Assume implementation for plot_tsne exists
        # plot_tsne(encoder, X_val, y_val)  # You may need to adjust this call based on actual implementation

        torch.save(encoder.state_dict(), args.encoder_path)
    else:
        if args.load_encoder:
            encoder.load_state_dict(torch.load(args.encoder_path))

        # Here you should convert your data into a DataLoader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
        fit_model(encoder, classifier, train_loader, val_loader, args)

        # For test predictions
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()
        test_loader = DataLoader(X_test, batch_size=args.batch_size)
        save_predictions_to_csv(encoder, classifier, test_loader, args)

def save_predictions_to_csv(encoder, classifier, data_loader, args):
    encoder.eval()
    classifier.eval()
    y_preds = []
    with torch.no_grad():
        for X_batch, in data_loader:
            X_batch = X_batch.to(ptu.device)
            embeddings = encoder(X_batch)
            y_pred_batch = classifier(embeddings)
            y_preds.extend(y_pred_batch.argmax(dim=1).cpu().numpy())
    np.savetxt(f'data/{args.sr_no}_predictions.csv', y_preds, delimiter=',', fmt='%d')
    print(f'Predictions saved to data/{args.sr_no}_predictions.csv')

if __name__ == "__main__":
    # Placeholder for argument parsing

# if __name__ == "__main__":
#     # Example: Parse command-line arguments and call main
#     args = Namespace(sr_no=12345, train_data_path="path/to/train_data", test_data_path="path/to/test_data", z_dim=128, batch_size=64, epochs=10, lr=1e-3, mode="cont_rep", load_encoder=False)
#     main(args)
