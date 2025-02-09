#ifndef REFUSION_TIMER_H
#define REFUSION_TIMER_H

#include <chrono>
#include <string>
#include <sstream>
#include <vector>

using namespace std;
using namespace chrono;

class Timer
{
public:
	Timer();
	~Timer() = default;

	void addMeasurement(string label);
	void addMeasurement(string label, int samples);
	string getTimingTrace();
private:
	std::vector<pair<string, time_point<high_resolution_clock>>> measurements {};
	std::vector<pair<string, int>> samples {};
};


#endif //REFUSION_TIMER_H
