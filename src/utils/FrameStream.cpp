#include "utils/FrameStream.h"
#include <fstream>
#include <iostream>

/**
 * Constructs the TUMVideo FrameStream based on a filepath to an associated.txt file.
 *
 * @param intrinsics the camera intrinsics for the footage.
 * @param associationDirectory the path to the directory containing the associated.txt file.
 * @param streamVideoFromDisk boolean indicating whether we should read all files in at instantiation or stream from the
 * 		disk during each call to nextFrame
 */
TUMVideo::TUMVideo(const string& associationDirectory, bool streamVideoFromDisk=false)
{
	cameraIntrinsics = RgbdSensor{525, 525, 319.5, 239.5, 5000, 480, 640};
	this->associationFilepath = associationDirectory + "/associated.txt";
	streamFromDisk = streamVideoFromDisk;
	frameCounter = 0;

	if(!streamFromDisk)
	{
		cout << "Inside TUM constructor, now reading in files" << endl;
		ifstream file {associationFilepath};
		cout << "parent path retrieved" << endl;
		string line;

		while(getline(file, line))
		{
			// The associated text file should be formatted in a particular way; we exploit this here.
			string rgbTimestamp, rgbFrame, depthTimestamp, depthFrame;
			istringstream(line) >> depthTimestamp >> depthFrame >> rgbTimestamp >> rgbFrame;

			// With the extracted filenames, the images are read and stored in the vectors.
			Frame frame {};

			cout << "Reading in frame " << associationDirectory + rgbFrame << endl;

			frame.rgb = cv::imread(associationDirectory + "/" + rgbFrame, CV_LOAD_IMAGE_COLOR);
			frame.depth = cv::imread(associationDirectory + "/" + depthFrame, CV_LOAD_IMAGE_ANYDEPTH);
			frame.depth.convertTo(frame.depth, CV_32FC1, 1.0 / 5000.0);

			frames.push_back(frame);
		}
	}
}

/**
 * Returns the next frame in the sequence.
 * TODO: For now this assumes that we are NOT streaming, and will exit the program if we attempt to stream.
 *
 * @return the next frame.
 */
Frame TUMVideo::nextFrame()
{
	if (this->streamFromDisk) { cout << "TUMVideo doesn't yet support disk streaming." << endl; exit(EXIT_FAILURE); }

	Frame frame = frames[frameCounter];
	frameCounter += 1;

	return frame;
}

/**
 * Returns the camera intrinsics.
 *
 * @return the camera intrinsics.
 */
RgbdSensor TUMVideo::getCameraIntrinsics()
{
	return cameraIntrinsics;
}

/**
 * Returns a bool indicating whether or not we have exhausted all of the available frames.
 *
 * @return indication boolean.
 */
bool TUMVideo::finished()
{
	return frameCounter == frames.size();
}

/**
 * Returns the number of frames in the video.
 *
 * @return the number of frames in the video.
 */
int TUMVideo::getFrameCount()
{
	return frames.size();
}