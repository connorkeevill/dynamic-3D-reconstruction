import os
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    path = "/app/data"
    directories = [d for d in listdir(path) if not isfile(join(path, d))]

    for directory in directories:
        print("-----------------------------------------------------------")
        print(f"Running on test dataset: {directory}")
        os.system(f"/app/catkin_ws/src/refusion/bin/refusion_example {join(path, directory)}")

    print("Complete")