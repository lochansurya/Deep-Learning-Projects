from argparse import Namespace

from utils import *
from LogisticRegression.model import LogisticRegression, SoftmaxRegression
from LogisticRegression.train_utils import fit_model


def main(args: Namespace):
    '''
    Main function to train and generate predictions in csv format

    Args:
    - args : Namespace : command line arguments
    '''
    # Get the training data
    X, y = get_data(args.train_data_path, is_linear=True, is_binary=args.mode == 'logistic')
    if args.gray_scale:
        X_train = convert_to_grayscale(X)
        X_train = X_train.reshape(X_train.shape[0], -1)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    
    # Create the model
    model = LogisticRegression(X.shape[1]) if args.mode == 'logistic'\
                else SoftmaxRegression(X.shape[1], len(np.unique(y)))

    # Train the model
    train_losses, train_accs, test_losses, test_accs = fit_model(
        model, X_train, y_train, X_val, y_val, num_iters=args.num_iters,
        lr=args.lr, batch_size=args.batch_size, l2_lambda=args.l2_lambda,
        grad_norm_clip=args.grad_norm_clip, is_binary=args.mode == 'logistic', args=args)

    # Plot the losses
    plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
    
    # Plot the accuracies
    plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
    
    # Get the test data
    X_test, _ = get_data(args.test_data_path, is_linear=True)

    # Save the predictions for the test data in a CSV file
    if args.mode == 'softmax':
        y_preds = []
        for i in range(0, len(X_test), args.batch_size):
            X_batch = X_test[i:i + args.batch_size]
            y_pred_batch = model(X_batch)
            y_preds.append(y_pred_batch)
        y_preds = np.concatenate(y_preds).argmax(axis=1)
        np.savetxt(f'data/{args.sr_no}_linear.csv', y_preds, delimiter=',', fmt='%d')
        print(f'Predictions saved to data/{args.sr_no}_linear.csv')
