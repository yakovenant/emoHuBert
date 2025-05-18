import os
import pathlib
from argparse import ArgumentTypeError


def dirpath_checker(path):
    if not os.path.exists(path):
        raise ArgumentTypeError(f'Directory {path} not found.')
    return path


def filepath_checker(path):
    if not pathlib.Path(path).parent.absolute().exists():
        raise ArgumentTypeError(f'File {path} not found.')
    return path


if __name__ == 'main':
    print('This is argument checker script.')
