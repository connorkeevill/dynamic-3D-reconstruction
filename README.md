# 3D Reconstruction of Dynamic Environments Using Egomotion Compensated Optical Flow

## Description
This system is developed as part of my undergraduate dissertation with the University of Bath, in a project which 
presents a 3D reconstruction system capable of handling dynamic environments and outperforming works from the state of 
the art.

This project is heavily adapted from the work of Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguère, Cyrill Stachniss'
paper "ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals", _arXiv_, 2019.

## Dependencies
I *strongly* advise the use of Docker to build this project. So much so that I'm not including a dependency list here.
If you choose not to use Docker, do so at your own peril...

However, there are some dependencies which are not covered by Docker. First and foremost, you will need a 
CUDA-enabled GPU.
This project has been tested on an Nvidia RTX 2080, but it should work on any Nvidia card which supports CUDA 9.0. YMMV.
You'll also need to have the necessary CUDA drivers for your card installed on your system.

You'll also need some RGB-D sequences to reconstruct. The system has been tested on the Bonn and TUM RGB-D datasets,
which are available [here](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) and 
[here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download), respectively.
I'd suggest using the Bonn dataset, as the system performs better on these sequences (for reasons explained in the dissertation).

The TUM dataset format (which these datasets both follow) represents the video sequences as two folders, one containing
the RGB frames and one containing the depth frames.
So, once you've downloaded the sequences, you'll also need to associate the depth and rgb frames with each other.
Fortunately, the TUM dataset provides a tool to do this (associate.py), which is available [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools).
You can also find this file in the /scripts folder of this repository, along with and associate-all.py script which will
run the associate.py script on all sequences in a folder.
Usage is as follows:
```
python associate-all.py [path/to/folder/with/sequences]
```
This will generate an associated.txt file for each sequence in the folder.

Finally, you'll need a config file. This is a TOML file which contains the parameters for the system.
This repository contains an example config file, which you can use as a template (which you can probably use as is).
The key parameters here are reconstruction_strategy, the [logging] section, and the [output] section.

## Building
The simplest this project is with Docker. I would strongly advise against building this project without Docker, as it has a lot of dependencies.

Building the project with Docker is simple.
First, clone the repository (if you don't already have a local copy of the code):
```
git clone git@github.com:connorkeevill/Dynamic_Scene_Reconstruction.git
git submodule init
git submodule update
```
Then, build the Docker image:
```
cd [project root]
docker build -t [image name] .
```
Go and make a cup of tea. This will take a while - this image builds OpenCV from source. Depending on your system this could take up to 2 hours.

Once this build is complete, you're all set.

## Running
To run the system, run the following command:
```
docker run --rm --gpus device=[gpu count] -v [path/to/folder/containing/sequences]:/app/data -v [config/file/path]:/app/config.toml [image name]
```
This will kick of a Python script (run.py), which will run the system on all sequences in the folder you specified.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License. See the LICENSE.txt file for details.
