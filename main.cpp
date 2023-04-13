#include <iostream>
#include <Eigen/Geometry>
#include "utils/FrameStream.h"
#include "tsdfvh/tsdf_volume.h"
#include "tracker/tracker.h"
#include <chrono>
#include <iomanip>
#include <settings.h>
#include <Logger.h>
#include <cpptoml.h>

using namespace std;
using namespace refusion;

int main(int argc, char** argv)
{
	// For now, config file will be in a fixed dir. Here we read in all the settings.
	shared_ptr<cpptoml::table> config = (cpptoml::parse_file("/app/config.toml"));

	// Use the read in settings to create the settings structs.
	settings settings = getSettings(*config);
	tsdfvh::TsdfVolumeOptions tsdf_options = getTsdfVolumeOptions(*config);
	TrackerOptions tracker_options = getTrackerOptions(*config);
	RgbdSensor sensor = getSensorConfig(*config);

	// Create the logger
	Logger logger {settings.verbose, settings.debug, settings.filepath};

	// Protect against no video being provided.
	if (argc == 0) {
		logger.error("No video provided. Ending.");
		exit(EXIT_FAILURE);
	}

	// Create TUMVideo object
	TUMVideo video {argv[1], settings.streamFrames};

	// Create tracker
	refusion::Tracker tracker {tsdf_options, tracker_options, sensor};

	// Create output file
	std::ofstream result((string)argv[1] + ".txt");

	// Start main loop
	while (!video.finished())
	{
		Frame frame = video.nextFrame();
		tracker.AddScan(frame.rgb, frame.depth);

		// Get the current pose
		Eigen::Matrix4d pose = tracker.GetCurrentPose();
		Eigen::Quaterniond rotation(pose.block<3, 3>(0, 0));

		// Write the pose to file
		result << std::fixed << std::setprecision(6) << video.getCurrentTimestamp()
			<< " " << pose.block<3, 1>(0, 3).transpose() << " "
			<< rotation.vec().transpose() << " " << rotation.w() << std::endl;
	}

	// Create mesh
	std::cout << "Creating mesh..." << std::endl;
	float3 low_limits = make_float3(-3, -3, 0);
	float3 high_limits = make_float3(3, 3, 4);
	refusion::tsdfvh::Mesh *mesh;
	cudaMallocManaged(&mesh, sizeof(refusion::tsdfvh::Mesh));
	*mesh = tracker.ExtractMesh(low_limits, high_limits);
	filepath_out.str("");
	filepath_out.clear();
	filepath_out << filebase << ".obj";
	mesh->SaveToFile(filepath_out.str());
}