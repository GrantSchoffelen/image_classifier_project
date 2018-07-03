
import argparse
from os import listdir


def main():
    in_arg = get_input_args()

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--top_k', type=int, default=3,
                        help='return top k most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='architecture model')
    parser.add_argument('--gpu', type=int, action='store_true',
                        help='use gpu')

    return parser.parse_args()


if __name__ == "__main__":
    main()
