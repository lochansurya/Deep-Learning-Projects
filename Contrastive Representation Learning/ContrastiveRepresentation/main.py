import torch
from argparse import Namespace
import gc
from torchsummary import summary
import matplotlib.pyplot as plt
import ContrastiveRepresentation.pytorch_utils as ptu
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
    print(f"args.batch_size: {args.batch_size}")

    # Set the seed
    torch.manual_seed(args.sr_no)

    # Get the training data
    X, y = get_data(args.train_data_path)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    
    # TODO: Convert the images and labels to torch tensors using pytorch utils (ptu)
    X= ptu.from_numpy(X)    # X will be a tensor and be on the GPU
    y= ptu.from_numpy(y)    # y will be a tensor and be on the GPU

    
    # instantiate the model
    encoder = Encoder(args.z_dim).to(ptu.device)
    # print the summary of the encoder
    summary(encoder, (3, 32, 32))
    if args.mode == 'fine_tune_linear':
        classifier = LinearClassifier(args.z_dim, 10)
        # classifier = # TODO: Create the linear classifier model
    elif args.mode == 'fine_tune_nn':
        classifier = Classifier(args.z_dim, 10)
    
    if args.mode == 'cont_rep':
        
        # Plot the t-SNE after fitting the encoder
        # get the embeddings from the encoder
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        contrastive_loss =[]
        X_train=ptu.from_numpy(X_train).float()
        y_train=ptu.from_numpy(y_train).float()
        contrastive_loss=(fit_contrastive_model(encoder, X_train, y_train, args.num_iters, args.batch_size, args.lr,args=args))
        print(f"mean of the contrastive loss: {np.mean(contrastive_loss)}")
        
        print(contrastive_loss)
        # Plot the contrastive loss
        # plot_contrep_losses(contrastive_loss, f'{args.mode} - Contrastive Losses')
        
        # Evaluate the embeddings in batches to avoid CUDA out of memory error
        batch_size = args.batch_size
        num_batches = len(X_train) // batch_size
        z = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if i == num_batches - 1:  # if this is the last batch
                end_idx = len(X_train)  # include all remaining samples
            X_batch = X_train[start_idx:end_idx].to(ptu.device)
            with torch.no_grad():
                z_batch = encoder(X_batch)
                z.append(z_batch)
                torch.cuda.empty_cache()
                gc.collect()
        z = torch.cat(z, dim=0)
        z = z.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()

        
        plot_tsne(z, y_train, args.batch_size)      
        # save the encoder
        torch.save(encoder.state_dict(), args.encoder_path)
    else: # train the classifier (fine-tune the encoder also when using NN classifier)
        # load the encoder
        torch.cuda.empty_cache()
        gc.collect()

        encoder.load_state_dict(torch.load(args.encoder_path)) 
        # Fit the model
        train_losses, train_accs, test_losses, test_accs = fit_model(
            encoder, classifier, X_train, y_train, X_val, y_val, args)
        
        # print the type fo the train_losses, train_accs, test_losses, test_accs
        print(f"type of train_losses: {type(train_losses)} | is instance of torch.Tensor: {isinstance(train_losses, torch.Tensor)} | len(train_losses): {len(train_losses)}")
        print(f"type of train_accs: {type(train_accs)} | is instance of torch.Tensor: {isinstance(train_accs, torch.Tensor)} | len(train_accs): {len(train_accs)}")
        print(f"type of test_losses: {type(test_losses)} | is instance of torch.Tensor: {isinstance(test_losses, torch.Tensor)} | len(test_losses): {len(test_losses)}")
        print(f"type of test_accs: {type(test_accs)} | is instance of torch.Tensor: {isinstance(test_accs, torch.Tensor)} | len(test_accs): {len(test_accs)}")

        # Plot the losses
        plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
        
        # Plot the accuracies
        plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
        
        # Get the test data
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()


        # Save the predictions for the test data in a CSV file
        y_preds = []
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
