#include "Timer.h"

Timer::Timer() {
	time_point<high_resolution_clock> start = high_resolution_clock::now();
	this->measurements.emplace_back("start", start);
}

void Timer::addMeasurement(string label) {
	time_point<high_resolution_clock> now = high_resolution_clock::now();
	this->measurements.emplace_back(label, now);
}

string Timer::getTimingTrace()
{
	stringstream stream {};

	auto total = duration_cast<microseconds>(measurements[measurements.size()].second - measurements[0].second).count();

	stream << "Total time: " << total / 1000000.0 << "s" << endl;

	for (int measurement = 1; measurement < this->measurements.size(); measurement++) {
		auto previous = this->measurements[measurement - 1];
		auto current = this->measurements[measurement];

		auto duration = duration_cast<microseconds>(current.second - previous.second).count();

		stream << previous.first << " -> " << current.first << ": " << duration / 1000000.0 << "s" << endl;
	}

	return stream.str();
}