import hashlib
import os

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    flatten = lambda l: [item for sublist in l for item in sublist]
    hashes = {}
    test_path = [os.path.join('data\\test\\raw', r) for r in os.listdir('data\\test\\raw')]
    train_path = [os.path.join('data\\train\\raw', r) for r in os.listdir('data\\train\\raw')]
    test_data = [[os.path.join(r, l) for l in os.listdir(r)] for r in test_path]
    train_data = [[os.path.join(r, l) for l in os.listdir(r)] for r in train_path]
    for file in flatten(test_data + train_data):
        if '.gitkeep' in file:
            continue

        hush = md5(file)
        if hush in hashes:
            print('Duplicate:', file)
        else:
            hashes[hush] = file

if __name__ == '__main__':
    main()
