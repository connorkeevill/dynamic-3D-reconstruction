#include "Timer.h"

Timer::Timer() {
	time_point<high_resolution_clock> start = high_resolution_clock::now();
	this->measurements.emplace_back("start", start);
}

void Timer::addMeasurement(string label) {
	time_point<high_resolution_clock> now = high_resolution_clock::now();
	this->measurements.emplace_back(label, now);
}

void Timer::addMeasurement(string label, int samples) {
	time_point<high_resolution_clock> now = high_resolution_clock::now();
	this->measurements.emplace_back(label, now);
	this->samples.emplace_back(label, samples);
}

string Timer::getTimingTrace()
{
	stringstream stream {};

	auto start = this->measurements[0].second;
	auto end = high_resolution_clock::now();

	auto total = duration_cast<microseconds>(end - start).count();

	stream << "Total; start -> now: " << total / 1000000.0 << "s" << endl;

	for (int measurement = 1; measurement < this->measurements.size(); measurement++) {
		auto previous = this->measurements[measurement - 1];
		auto current = this->measurements[measurement];

		auto duration = duration_cast<microseconds>(current.second - previous.second).count();

		stream << previous.first << " -> " << current.first << ": " << duration / 1000000.0 << "s" << endl;

		// If the label is in samples, we want to output the time divide by the number of samples:
		for (auto sample : this->samples) {
			if (sample.first == current.first) {
				stream << previous.first << " -> " << current.first << " per sample: " << duration / 1000000.0 / sample.second << "s" << endl;
			}
		}
	}

	return stream.str();
}