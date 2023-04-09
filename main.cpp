#include <iostream>
#include <Eigen/Geometry>
#include <opencl-c-base.h>
#include "utils/FrameStream.h"
#include "tsdfvh/tsdf_volume.h"
#include "tracker/tracker.h"

using namespace std;

refusion::Tracker createTracker()
{
	// Options for the TSDF representation
	refusion::tsdfvh::TsdfVolumeOptions tsdf_options{};
	tsdf_options.voxel_size = 0.01;
	tsdf_options.num_buckets = 50000;
	tsdf_options.bucket_size = 10;
	tsdf_options.num_blocks = 500000;
	tsdf_options.block_size = 8;
	tsdf_options.max_sdf_weight = 64;
	tsdf_options.truncation_distance = 0.1;
	tsdf_options.max_sensor_depth = 5;
	tsdf_options.min_sensor_depth = 0.1;

	// Options for the tracker
	refusion::TrackerOptions tracker_options;
	tracker_options.max_iterations_per_level[0] = 6;
	tracker_options.max_iterations_per_level[1] = 3;
	tracker_options.max_iterations_per_level[2] = 2;
	tracker_options.downsample[0] = 4;
	tracker_options.downsample[1] = 2;
	tracker_options.downsample[2] = 1;
	tracker_options.min_increment = 0.0001;
	tracker_options.regularization = 0.002;
	tracker_options.huber_constant = 0.02;
	tracker_options.remove_dynamics = false;

	// Intrinsic parameters of the sensor
	refusion::RgbdSensor sensor{};
	sensor.cx = 319.5f;
	sensor.cy = 239.5f;
	sensor.fx = 525.0;
	sensor.fy = 525.0;
	sensor.rows = 480;
	sensor.cols = 640;
	sensor.depth_factor = 5000;

	return refusion::Tracker{tsdf_options, tracker_options, sensor};
}

int main(int argc, char** argv)
{
	auto start = chrono::high_resolution_clock::now();

	if (argc == 0) {
		cerr << "No file provided, ending" << endl;
		exit(0);
	}


	string filepath = argv[1];
	cout << "Received filepath: " << filepath << endl;

	cout << "Reading frames from disk..." << endl;
	TUMVideo video = TUMVideo{filepath, false};
	cout << "Frames read." << endl;

	auto frames_read = chrono::high_resolution_clock::now();

	refusion::Tracker tracker = createTracker();

	std::string filebase(argv[1]);
	std::stringstream filepath_out, filepath_time;
	filepath_out << filebase << "/result.txt";
	std::ofstream result(filepath_out.str());

	auto before_loop = chrono::high_resolution_clock::now();

	while(!video.finished()) {
		Frame frame = video.nextFrame();
		tracker.AddScan(frame.rgb, frame.depth);
//		Eigen::Matrix4d pose = tracker.GetCurrentPose();
//		Eigen::Quaterniond rotation(pose.block<3, 3>(0, 0));
//		std::cout << tracker.GetCurrentPose() << std::endl;

//		cv::Mat virtual_rgb = tracker.GenerateRgb(1280, 960);
	}

	auto after_loop = chrono::high_resolution_clock::now();

	std::cout << "Creating mesh..." << std::endl;
	float3 low_limits = make_float3(-3, -3, 0);
	float3 high_limits = make_float3(3, 3, 4);
	refusion::tsdfvh::Mesh *mesh;
	cudaMallocManaged(&mesh, sizeof(refusion::tsdfvh::Mesh));
	*mesh = tracker.ExtractMesh(low_limits, high_limits);
	filepath_out.str("");
	filepath_out.clear();
	filepath_out << filebase << "mesh.obj";
	mesh->SaveToFile(filepath_out.str());

	auto after_mesh = chrono::high_resolution_clock::now();

	auto readFrames = chrono::duration_cast<chrono::microseconds>(frames_read - start);
	auto loop = chrono::duration_cast<chrono::microseconds>(after_loop - before_loop);
	auto mesh_time = chrono::duration_cast<chrono::microseconds>(after_mesh - after_loop);
	auto total = chrono::duration_cast<chrono::microseconds>(after_mesh - start);

	cout << "Total time: " << total.count() / 1000000.0 << "s" << endl;
	cout << "Read frames: " << readFrames.count() / 1000000.0 << "s" << endl;
	cout << "Loop: " << loop.count() / 1000000.0 << "s" << endl;
	cout << "Time per frame: " << loop.count() / 1000000.0 / video.getFrameCount() << "s" << endl;
	cout << "Mesh: " << mesh_time.count() / 1000000.0 << "s" << endl;

	return EXIT_SUCCESS;
}