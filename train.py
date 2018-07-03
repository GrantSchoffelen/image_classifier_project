import argparse
from os import listdir


def main():
    in_arg = get_input_args()

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg',
                        help='architecture model')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='hidden units')
    parser.add_argument('--epochs', type=int, default=3,
                        help='epochs')
    parser.add_argument('--epochs', type=int, default=3,
                        help='epochs')
    parser.add_argument('--gpu', type=int, action='store_true',
                        help='use gpu')

    return parser.parse_args()


if __name__ == "__main__":
    main()
