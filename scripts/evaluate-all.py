import os
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    path = "."
    directories = [d for d in listdir(path) if not isfile(join(path, d))]

    for directory in directories:
        os.system(f"python3 evaluate_ate.py --plot {directory + '-ate'}.png {join(path, directory, 'groundtruth.txt')} {join(path, directory + '.txt')}")
        os.system(f"python3 evaluate_rpe.py --plot {directory + '-rpe'}.png --fixed_delta {join(path, directory, 'groundtruth.txt')} {join(path, directory + '.txt')}")

    print("All Done")