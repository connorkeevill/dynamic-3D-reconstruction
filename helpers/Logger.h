
#ifndef REFUSION_LOGGER_H
#define REFUSION_LOGGER_H

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;


class Logger
{
public:
	Logger(bool verbose, bool debug);
	Logger(bool verbose, bool debug, string filepath);
	~Logger() = default;

	void alwaysLog(string message);
	void verboseLog(string message);
	void debugLog(string message);
	void error(string message);
private:
	bool verbose;
	bool debug;
	bool writeToFile;
	ofstream fileStream;

	void consoleLog(string message);
	void fileLog(string message);
};


#endif //REFUSION_LOGGER_H
