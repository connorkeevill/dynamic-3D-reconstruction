#ifndef REFUSION_TIMER_H
#define REFUSION_TIMER_H

#include <chrono>
#include <string>
#include <sstream>

using namespace std;
using namespace chrono;

class Timer
{
public:
	Timer();
	~Timer() = default;

	void addMeasurement(string label);
	string getTimingTrace();
private:
	vector<pair<string, time_point<high_resolution_clock>>> measurements;
};


#endif //REFUSION_TIMER_H
