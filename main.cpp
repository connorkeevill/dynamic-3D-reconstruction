#include <iostream>
#include <Eigen/Geometry>
#include "utils/FrameStream.h"
#include "tsdfvh/tsdf_volume.h"
#include "tracker/tracker.h"
#include <iomanip>
#include <settings.h>
#include <Logger.h>
#include <cpptoml.h>
#include <Timer.h>

using namespace std;
using namespace refusion;

int main(int argc, char** argv)
{
	// Initially a new line to separate from any previous output.
	cout << endl;

	Timer timer {};

	// For now, config file will be in a fixed dir. Here we read in all the settings.
	shared_ptr<cpptoml::table> config = cpptoml::parse_file("/app/config.toml");

	// Use the read in settings to create the settings structs.
	settings settings = getSettings(*config);
	tsdfvh::TsdfVolumeOptions tsdf_options = getTsdfVolumeOptions(*config);
	TrackerOptions tracker_options = getTrackerOptions(*config);
	RgbdSensor sensor = getSensorConfig(*config);

	// Create the logger
	Logger *logger = new Logger(settings.verbose, settings.debug);

	logger->alwaysLog("Settings loaded. Starting reconstruction...");
	logger->verboseLog("VERBOSITY ON");
	logger->debugLog("DEBUG ON");
	timer.addMeasurement("Settings loaded");

	// Protect against no video being provided.
	if (argc == 0) {
		logger->error("No video provided. Ending.");
		exit(EXIT_FAILURE);
	}

	logger->verboseLog("File provided: " + (string)argv[1] + ".");

	// Create TUMVideo object
	logger->verboseLog("Reading frames...");
	TUMVideo video {argv[1], settings.streamFrames};
	logger->verboseLog("Frames read.");
	timer.addMeasurement("Frames read");

	// Create tracker
	logger->verboseLog("Creating tracker...");
	refusion::Tracker tracker {tsdf_options, tracker_options, sensor, logger};
	logger->verboseLog("Tracker created.");
	timer.addMeasurement("Tracker created");

	// Create output file
	std::ofstream result((string)argv[1] + ".txt");
	logger->verboseLog("Output file created.");

	// Start main loop
	logger->verboseLog("Starting main loop...");
	while (!video.finished())
	{
		logger->debugLog("Processing frame " + to_string(video.getCurrentFrameIndex()) + ".");

		Frame frame = video.nextFrame();
		logger->debugLog("Frame received from video stream.");

		tracker.AddScan(frame.rgb, frame.depth);
		logger->debugLog("Frame added to tracker.");

		if(settings.outputResults)
		{
			// Get the current pose
			Eigen::Matrix4d pose = tracker.GetCurrentPose();
			Eigen::Quaterniond rotation(pose.block<3, 3>(0, 0));

			// Write the pose to file
			result << std::fixed << std::setprecision(6) << video.getCurrentTimestamp()
				<< " " << pose.block<3, 1>(0, 3).transpose() << " "
				<< rotation.vec().transpose() << " " << rotation.w() << std::endl;
		}
	}
	logger->verboseLog("Main loop finished.");
	timer.addMeasurement("Main loop finished");

	if (settings.outputMesh)
	{
		// Create mesh
		logger->verboseLog("Saving mesh...");
		float3 low_limits = make_float3(-3, -3, 0);
		float3 high_limits = make_float3(3, 3, 4);
		refusion::tsdfvh::Mesh *mesh;
		cudaMallocManaged(&mesh, sizeof(refusion::tsdfvh::Mesh));
		*mesh = tracker.ExtractMesh(low_limits, high_limits);
		mesh->SaveToFile((string)argv[1] + ".obj");

		logger->verboseLog("Mesh saved.");
		timer.addMeasurement("Mesh saved");
	}


	logger->alwaysLog(timer.getTimingTrace());
	logger->alwaysLog("");  // New line
	logger->alwaysLog("Done.");

	return EXIT_SUCCESS;
}