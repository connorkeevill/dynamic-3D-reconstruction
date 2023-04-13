
#ifndef REFUSION_LOGGER_H
#define REFUSION_LOGGER_H

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <map>

using namespace std;
using namespace cv;


class Logger
{
public:
	Logger(bool verbose, bool debug, string filebase);
	Logger(bool verbose, bool debug, string filebase, string filepath);
	~Logger() = default;

	void alwaysLog(string message);
	void verboseLog(string message);
	void debugLog(string message);
	void error(string message);
	void addFrameToOutputVideo(Mat frame, string videoName);

	void release();
private:
	bool verbose;
	bool debug;
	string filebase;
	bool writeToFile;
	ofstream fileStream;
	map<string, VideoWriter> outputVideos;

	void consoleLog(string message);
	void fileLog(string message);
	void createNewVideo(string videoName, Mat frame);
};


#endif //REFUSION_LOGGER_H
