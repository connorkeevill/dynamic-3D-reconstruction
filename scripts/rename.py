import os
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    path = "."
    directories = [d for d in listdir(path) if not isfile(join(path, d))]

    for directory in directories:
        print(f"Moving file {join(directory, 'mesh.obj')} to {directory + '.obj'}")
        os.system(f"mv {join(path, directory, 'mesh.obj')} {directory + '.obj'}")
