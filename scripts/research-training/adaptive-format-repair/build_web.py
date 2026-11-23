"""Assemble the static browser demo with the exact Python engine and frozen corpus."""
import argparse
from hashlib import sha256
from pathlib import Path
import shutil

from benchmark import CORPUS_SHA256

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='new or empty static output directory')
    args = parser.parse_args()
    destination = args.destination.resolve()
    assets = HERE / 'web'
    if destination.is_relative_to(assets):
        parser.error('destination must be outside the web source directory')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error('destination must be a new or empty directory')
    if not (assets / 'index.html').is_file():
        parser.error('web/index.html is missing')
    corpus = HERE / 'CASES.json'
    if sha256(corpus.read_bytes()).hexdigest() != CORPUS_SHA256:
        parser.error('CASE corpus SHA-256 does not match')
    shutil.copytree(assets, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(HERE / 'repair.py', destination / 'repair.py')
    shutil.copyfile(corpus, destination / 'CASES.json')


if __name__ == '__main__':
    main()
