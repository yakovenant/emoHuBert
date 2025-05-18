import os
import argparse
import pandas as pd

from .inference import make_predicts
from .train import classification_train, triplet_train
from .test import test_model
from .utils import args_checker


def predict_mode(args):
    predictions = make_predicts(args.dir, args.model)
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(args.output)
    print(f'Results are saved to {args.output}')


def train_mode(args):
    if args.loss == 'CrossEntropy':
        classification_train(
            filepath=args.file,
            dirpath=args.dir,
            output_dir=args.output,
            model_dir=args.model,
            n_labels=args.n_train_classes,
            n_epochs=args.n_train_epochs,
            batch_size=args.n_train_batches,
            learning_rate=args.init_learning_rate,
            grad_accum_steps=args.n_grad_accum_steps,
            device=args.device)
    elif args.loss == 'Triplet':
        triplet_train(
            filepath=args.file,
            dirpath=args.dir,
            output_dir=args.output,
            model_dir=args.model,
            n_epochs=args.n_train_epochs,
            batch_size=args.n_train_batches,
            grad_accum_steps=args.n_grad_accum_steps,
            device=args.device)
    else:
        raise ValueError()


def test_mode(args):
    test_model(args.file, args.dir, args.model, args.output)


if __name__ == 'main':
    if 1: # for debug
        # Mode
        run_mode = 'train'  # train test predict
        # Predict defaults
        path_to_predict_wav = os.path.normpath('...')
        path_to_classifier = os.path.normpath('...')
        name_results_file = 'results.csv'
        n_predict_classes = 2

        # Train defaults
        path_to_train_wav = os.path.normpath('...')
        path_to_train_csv = os.path.normpath('...')
        path_to_pretrained_model = ''  # facebook/hubert-base-ls960
        name_finetuned_model = 'hubert-ft'
        name_loss = 'CrossEntropy'  # CrossEntropy Triplet
        n_train_classes = 2
        n_train_epochs = 5
        n_train_batches = 2
        n_grad_accum_steps = 2
        init_learning_rate = 5e-5
        device = 'gpu'  # cpu gpu

        # Test defaults
        path_to_test_wav = os.path.normpath('...')
        path_to_test_csv = os.path.normpath('...')
        name_metrics_file = 'metrics.json'

    # Init parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Operating mode')

    # Predict parser args
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('-d', '--dir', type=args_checker.dirpath_checker, # required=True,
                                default=path_to_predict_wav,
                                help='Path to audio data dir')
    parser_predict.add_argument('-m', '--model', type=args_checker.dirpath_checker, # required=True,
                                default=path_to_classifier,
                                help='Path to classification model dir')
    parser_predict.add_argument('-o', '--output', type=args_checker.filepath_checker,
                                default=name_results_file,
                                help='Name of file with classification results')
    parser_predict.add_argument('-c', choices=[2],
                                default=n_predict_classes,
                                help='Number of emotion types to classify')
    parser_predict.set_defaults(func=predict_mode)

    # Train parser args
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('-d', type=args_checker.dirpath_checker, # required=True,
                              default=path_to_train_wav,
                              help='Path to train audio data dir')
    parser_train.add_argument('-f', type=args_checker.filepath_checker, # required=True,
                              default=path_to_train_csv,
                              help='Path to .csv file with data labels')
    parser_train.add_argument('-m',  type=args_checker.filepath_checker,
                              default=path_to_pretrained_model,
                              help='Path to baseline pretrained model')
    parser_train.add_argument('-c', choices=[2],
                              default=n_train_classes,
                              help='Number of emotion types to classify')
    parser_train.add_argument('-o', type=args_checker.filepath_checker,
                              default=name_finetuned_model,
                              help='Name of output fine-tuned model dir')
    parser_train.add_argument('-e', type=int,
                              default=n_train_epochs,
                              help='Number of training epochs')
    parser_train.add_argument('-b', type=int,
                              default=n_train_batches,
                              help='Size of training batch')
    parser_train.add_argument('-g', type=int,
                              default=n_grad_accum_steps,
                              help='Number of gradient accumulation steps')
    parser_train.add_argument('-lr', type=float,
                              default=init_learning_rate,
                              help='Initial learning rate value')
    parser_train.add_argument('-loss', choises=['CrossEntropy', 'Triplet'],
                              default=name_loss,
                              help='Loss function type')
    parser_train.add_argument('--device', choices=['cpu', 'gpu'],
                              default=device,
                              help='Training device environment')
    parser_train.set_defaults(func=train_mode)

    # Test parser args
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('-d', '--dir', type=args_checker.dirpath_checker, # required=True,
                             default=path_to_test_wav,
                             help='Path to test audio dir')
    parser_test.add_argument('-f', '--file', type=args_checker.filepath_checker, # required=True,
                             default=path_to_test_csv,
                             help='Path to .csv file with data labels')
    parser_test.add_argument('-m', '--model', type=args_checker.dirpath_checker, # required=True,
                             default=path_to_classifier,
                             help='Path to classification model dir')
    parser_test.add_argument('-o', '--output', type=args_checker.filepath_checker,
                             default=name_metrics_file,
                             help='Name of .json file to save metrics estimation')
    parser_test.set_defaults(func=test_mode)

    #args = parser.parse_args()
    if run_mode == 'predict':
        args = parser_predict.parse_args()
    elif run_mode == 'train':
        args = parser_train.parse_args()
    elif run_mode == 'test':
        args = parser_test.parse_args()
    else:
        raise ValueError()

    if not vars(args):
        parser.print_help()
    else:
        args.func(args)
