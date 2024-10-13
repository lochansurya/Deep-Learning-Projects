from argparse import ArgumentParser

from utils import *
from LogisticRegression.model import LogisticRegression, SoftmaxRegression
from LogisticRegression.train_utils import fit_model, evaluate_model



if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser(description='E0-270 Assignment 1')
    parser.add_argument(
        'sr_no', type=int, default=0,
        help='Your 5 digit SR Number')
    parser.add_argument(
        '--train_data_path', type=str, default="data/cifar10_train.npz",
        help='Path to the training data')
    parser.add_argument(
        '--test_data_path', type=str, default="data/test_images.npz",
        help='Path to the test data')
    parser.add_argument(
        '--num_iters', type=int, default=1000,
        help='Number of iterations for training')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for training')
    parser.add_argument(
        '--mode', type=str, default='logistic', choices=
        ['logistic', 'softmax', 'cont_rep', 'fine_tune_linear', 'fine_tune_nn'],
        help='Mode of operation'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Batch size for training')
    parser.add_argument(
        '--l2_lambda', type=float, default=0.1,
        help='L2 regularization for training')
    parser.add_argument(
        '--grad_norm_clip', type=float, default=4.0,
        help='Clip gradient norm')
    # Contrastive Representation Learning specific arguments
    parser.add_argument(
        '--z_dim', type=int, default=32,
        help='Representation dimension for the encoder')
    parser.add_argument(
        '--encoder_path', type=str, default='models/encoder.pth',
        help='Path to save the encoder model')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--gray_scale', type= bool, default=False,
                        help='Convert images to grayscale for preprocessing')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum parameter for parameter updates')
    args = parser.parse_args()
    args.test_data_path = args.test_data_path[:-4] + f'_{args.sr_no}.npz'

    # set the seed
    np.random.seed(args.sr_no)

    # lazy loading of the main function depending on the mode
    if args.mode == 'logistic' or args.mode == 'softmax':
        from LogisticRegression.main import main
    elif args.mode == 'cont_rep' or args.mode == 'fine_tune_linear' or args.mode == 'fine_tune_nn':
        from ContrastiveRepresentation.main import main
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    
    assert args.sr_no > 10000 and args.sr_no < 30000,\
        'You must enter your 5 digit SR Number'

    # Run the main function
    main(args)
