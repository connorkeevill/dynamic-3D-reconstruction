import os
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    path = "."
    directories = [d for d in listdir(path) if not isfile(join(path, d))]

    for directory in directories:
        os.system(f"python3 associate.py {directory}/depth.txt {directory}/rgb.txt > {directory}/associated.txt")

    print("All Done")