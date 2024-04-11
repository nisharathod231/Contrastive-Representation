import torch
import ContrastiveRepresentation.pytorch_utils as ptu
from argparse import Namespace
from utils import *
from LogisticRegression.model import SoftmaxRegression as LinearClassifier
from ContrastiveRepresentation.model import Encoder, Classifier
from ContrastiveRepresentation.train_utils import fit_contrastive_model, fit_model


def main(args: Namespace):
    '''
    Main function to train and generate predictions in csv format

    Args:
    - args : Namespace : command line arguments
    '''
    # Set the seed
    torch.manual_seed(args.sr_no)

    # Get the training data
    X, y = get_data(args.train_data_path)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    classes = len(np.unique(y_train))

    # TODO: Convert the images and labels to torch tensors using pytorch utils (ptu)
    
    # Create the model
    encoder = Encoder(args.z_dim).to(ptu.device)
    if args.mode == 'fine_tune_linear':
        # TODO: Create the linear classifier model
        classifier = LinearClassifier(args.z_dim, classes)
    elif args.mode == 'fine_tune_nn':
        # TODO: Create the neural network classifier model
        classifier = Classifier(args.z_dim, classes)
        classifier = classifier.to(ptu.device)
    
    if args.mode == 'cont_rep':
        #raise NotImplementedError('Implement the contrastive representation learning')
        # X_train = ptu.from_numpy(X_train)
        # y_train = ptu.from_numpy(y_train)
        # X_val = ptu.from_numpy(X_val)
        X_train = ptu.from_numpy(X_train).float()
        y_train = ptu.from_numpy(y_train).float()
        X_val = ptu.from_numpy(X_val).float()
        encoder.load_state_dict(torch.load('models/enocoder.pth'))
        # save the encoder
        #10m->0.45823
        crloss = fit_contrastive_model(encoder, X_train, y_train, num_iters=args.num_iters, batch_size=args.batch_size, learning_rate=args.lr)
        print("Saving encoder...")
        torch.save(encoder.state_dict(), args.encoder_path)
        print("Encoder is saved at models/encoder.pth")
        encoder.eval()
        z = None

        for lv in range(0, len(X_val), args.batch_size):
            tot = args.batch_size + lv
            if tot > len(X_val):
                # X_batch = X_val[lv:]
                X_batch = X_val[lv:].to(ptu.device)
            else:
                # X_batch = X_val[lv:tot]
                X_batch = X_val[lv:tot].to(ptu.device)
            crb = encoder(X_batch)
            crb = ptu.to_numpy(crb)
            if z is None:
                z = crb
            else:
                z = np.concatenate((z, crb))   
        # Plot the t-SNE after fitting the encoder         
        plot_tsne(z, y_val) #axis chanfe karni hai
    else: # train the classifier (fine-tune the encoder also when using NN classifier)
        # load the encoder
        # raise NotImplementedError('Load the encoder')
        encoder.load_state_dict(torch.load('models/encoder.pth'))
        X_train = ptu.from_numpy(X_train)
        y_train = ptu.from_numpy(y_train, torch.long)

        X_val = ptu.from_numpy(X_val)
        y_val = ptu.from_numpy(y_val, torch.long)
        
        # Fit the model
        train_losses, train_accs, test_losses, test_accs = fit_model(
            encoder, classifier, X_train, y_train, X_val, y_val, args)
        
        # Plot the losses
        plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
        
        # Plot the accuracies
        plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
        
        # Get the test data
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()

        # Save the predictions for the test data in a CSV file
        # var = 0
        encoder.eval()
        if args.mode == 'fine_tune_nn': classifier.eval()
        y_preds = [] #1000:256:0.37190
        for i in range(0, len(X_test), args.batch_size):
            X_batch = X_test[i:i+args.batch_size].to(ptu.device)
            repr_batch = encoder(X_batch)
            if 'linear' in args.mode:
                repr_batch = ptu.to_numpy(repr_batch)
            y_pred_batch = classifier(repr_batch)
            if 'nn' in args.mode:
                y_pred_batch = ptu.to_numpy(y_pred_batch)
            y_preds.append(y_pred_batch)
        y_preds = np.concatenate(y_preds).argmax(axis=1)
        np.savetxt(f'data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv',\
                y_preds, delimiter=',', fmt='%d')
        print(f'Predictions saved to data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv')
