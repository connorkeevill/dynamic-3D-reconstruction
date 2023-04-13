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
