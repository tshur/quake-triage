"""Differ: Outputs duplicate files that we downloaded on accident."""

import hashlib
import os

def md5(fname):
    """Calculates the md5 sum of a file."""

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    # helper function for flattening array of arrays
    flatten = lambda l: [item for sublist in l for item in sublist]

    hashes = {}

    # get all files in train/test and label subdirectories
    test_path = [os.path.join('data\\test\\raw', r) for r in os.listdir('data\\test\\raw')]
    train_path = [os.path.join('data\\train\\raw', r) for r in os.listdir('data\\train\\raw')]
    test_data = [[os.path.join(r, l) for l in os.listdir(r)] for r in test_path]
    train_data = [[os.path.join(r, l) for l in os.listdir(r)] for r in train_path]

    # go through each subdirectory file (ignore gitkeeps)
    for file in flatten(test_data + train_data):
        if '.gitkeep' in file:
            continue

        # if hash in map, then duplicate; otherwise add hash to map
        hash_ = md5(file)
        if hash_ in hashes:
            print('Duplicate:', file, 'with:', hashes[hash_])
        else:
            hashes[hash_] = file

if __name__ == '__main__':
    main()
