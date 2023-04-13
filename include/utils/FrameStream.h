#ifndef DYNAMIC_SCENE_RECONSTRUCTION_FRAMESTREAM_H
#define DYNAMIC_SCENE_RECONSTRUCTION_FRAMESTREAM_H

#include "utils/rgbd_sensor.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace refusion;

struct Frame {
    cv::Mat rgb;
    cv::Mat depth;
};

class FrameStream
{
public:
	virtual ~FrameStream() = default;

	virtual RgbdSensor getCameraIntrinsics() { return RgbdSensor{}; };
	virtual Frame nextFrame() { return Frame{}; };
	virtual bool finished() { return true; };
};

class TUMVideo : public FrameStream
{
public:
	TUMVideo(const string&, bool);

	Frame nextFrame() override;
	RgbdSensor getCameraIntrinsics() override;
	bool finished() override;
	int getFrameCount();
	int getCurrentFrameIndex();
	double getCurrentTimestamp();
private:
	vector<Frame> frames{};
	vector<double> timestamps{};

	RgbdSensor cameraIntrinsics;
	string associationFilepath;
	bool streamFromDisk;
	int frameCounter;
};


#endif //DYNAMIC_SCENE_RECONSTRUCTION_FRAMESTREAM_H
