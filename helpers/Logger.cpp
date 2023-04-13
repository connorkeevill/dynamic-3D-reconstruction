//
// Created by Connor Keevill on 13/04/2023.
//

#include "Logger.h"

/**
 * Constructor.
 *
 * @param verbose verbosity flag
 * @param debug debug flag
 * @param writeToFile write to file flag
 */
Logger::Logger(bool verbose, bool debug) {
	this->verbose = verbose;
	this->debug = debug;
}

/**
 * Overloaded constructor allowing a filepath to be given.
 *
 * @param verbose verbosity flag
 * @param debug debug flag
 * @param filepath the filepath to write to
 */
Logger::Logger(bool verbose, bool debug, string filepath) : Logger(verbose, debug)
{
	this->writeToFile = true;
	fileStream.open(filepath + ".txt");
}

/**
 * (Always) logs the given message.
 *
 * @param message the message to log.
 */
void Logger::alwaysLog(string message) {
	consoleLog(message);
	fileLog(message);
}

/**
 * Logs the given message if verbosity set to true.
 *
 * @param message the message to log.
 */
void Logger::verboseLog(string message) {
	if (verbose) {
		consoleLog(message);
		fileLog(message);
	}
}

/**
 * Logs the given message if debug set to true.
 *
 * @param message the message to log.
 */
void Logger::debugLog(string message) {
	if (debug) {
		consoleLog(message);
		fileLog(message);
	}
}

/**
 * Logs the given message to the error stream.
 *
 * @param message the message to log.
 */
void Logger::error(string message) {
	cerr << message << endl;
}

/**
 * Adds the given frame to the output video with the given name.
 * If the video does not exist, it will be created.
 *
 * @param frame the frame to add to the video.
 * @param videoName the name of the video to add the frame to.
 */
void Logger::addFrameToOutputVideo(Mat frame, string videoName) {
	if(!outputVideos.count(videoName)) {  // If the video is in the map, the count will be 1. Otherwise it will be 0.
		createNewVideo(videoName, frame);
	}

	outputVideos[videoName].write(frame);
}

/**
 * Releases all the videos.
 */
void Logger::release() {
	for (auto& video : outputVideos) {
		video.second.release();
	}
}

/**
 * Logs the given message to the console.
 *
 * @param message the message to log.
 */
void Logger::consoleLog(string message) {
	cout << message << endl;
}

/**
 * Logs the given message to the file.
 *
 * @param message the message to log.
 */
void Logger::fileLog(string message)
{
	if (writeToFile) {
		fileStream << message << endl;
	}
}

/**
 * Creates a new video with the given name and frame size.
 *
 * @param videoName the name of the video to create.
 * @param frame the frame to use to determine the size of the video.
 */
void Logger::createNewVideo(string videoName, Mat frame) {
	VideoWriter videoWriter;
	videoWriter.open(videoName + ".avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, frame.size());
	outputVideos[videoName] = videoWriter;
}
