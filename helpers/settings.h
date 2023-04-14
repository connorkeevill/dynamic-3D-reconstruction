//
// Created by Connor Keevill on 13/04/2023.
//

#ifndef REFUSION_SETTINGS_H
#define REFUSION_SETTINGS_H

#include <cpptoml.h>
#include "tracker/tracker.h"
#include "tsdfvh/tsdf_volume.h"

using namespace refusion;

struct settings
{
	/** Logging options **/
	bool verbose;
	bool debug;
	bool writeLogsToFile;
	string filepath;

	/** Input Options **/
	bool streamFrames;

	/** Output options **/
	bool outputMesh;
	bool outputResults;
	bool outputTimings;
	bool outputReprojectedVideo;
};

/**
 * Loads the tsdf settings from the given config table and returns them.
 *
 * @param config the config file
 * @return the settings
 */
tsdfvh::TsdfVolumeOptions getTsdfVolumeOptions(const cpptoml::table& config)
{
	refusion::tsdfvh::TsdfVolumeOptions tsdf_options{};
	tsdf_options.voxel_size = config.get_qualified_as<double>("tsdf_options.voxel_size").value_or(tsdf_options.voxel_size);
	tsdf_options.num_buckets = config.get_qualified_as<int>("tsdf_options.num_buckets").value_or(tsdf_options.num_buckets);
	tsdf_options.bucket_size = config.get_qualified_as<int>("tsdf_options.bucket_size").value_or(tsdf_options.bucket_size);
	tsdf_options.num_blocks = config.get_qualified_as<int>("tsdf_options.num_blocks").value_or(tsdf_options.num_blocks);
	tsdf_options.block_size = config.get_qualified_as<int>("tsdf_options.block_size").value_or(tsdf_options.block_size);
	tsdf_options.max_sdf_weight = config.get_qualified_as<int>("tsdf_options.max_sdf_weight").value_or(tsdf_options.max_sdf_weight);
	tsdf_options.truncation_distance = config.get_qualified_as<double>("tsdf_options.truncation_distance").value_or(tsdf_options.truncation_distance);
	tsdf_options.max_sensor_depth = config.get_qualified_as<double>("tsdf_options.max_sensor_depth").value_or(tsdf_options.max_sensor_depth);
	tsdf_options.min_sensor_depth = config.get_qualified_as<double>("tsdf_options.min_sensor_depth").value_or(tsdf_options.min_sensor_depth);

	return tsdf_options;
}

/**
 * Loads the tracker settings from the given config table and returns them.
 *
 * @param config the config file
 * @return the settings
 */
TrackerOptions getTrackerOptions(const cpptoml::table &config)
{
	refusion::TrackerOptions tracker_options {};
	tracker_options.max_iterations_per_level[0] = config.get_qualified_as<int>("tracker_options.max_iterations_per_level_0").value_or(tracker_options.max_iterations_per_level[0]);
	tracker_options.max_iterations_per_level[1] = config.get_qualified_as<int>("tracker_options.max_iterations_per_level_1").value_or(tracker_options.max_iterations_per_level[1]);
	tracker_options.max_iterations_per_level[2] = config.get_qualified_as<int>("tracker_options.max_iterations_per_level_2").value_or(tracker_options.max_iterations_per_level[2]);
	tracker_options.downsample[0] = config.get_qualified_as<int>("tracker_options.downsample_0").value_or(tracker_options.downsample[0]);
	tracker_options.downsample[1] = config.get_qualified_as<int>("tracker_options.downsample_1").value_or(tracker_options.downsample[1]);
	tracker_options.downsample[2] = config.get_qualified_as<int>("tracker_options.downsample_2").value_or(tracker_options.downsample[2]);
	tracker_options.min_increment = config.get_qualified_as<double>("tracker_options.min_increment").value_or(tracker_options.min_increment);
	tracker_options.regularization = config.get_qualified_as<double>("tracker_options.regularization").value_or(tracker_options.regularization);
	tracker_options.huber_constant = config.get_qualified_as<double>("tracker_options.huber_constant").value_or(tracker_options.huber_constant);
	tracker_options.remove_dynamics = config.get_qualified_as<bool>("tracker_options.remove_dynamics").value_or(tracker_options.remove_dynamics);
	tracker_options.output_mask_video = config.get_qualified_as<bool>("tracker_options.output_mask_video").value_or(tracker_options.output_mask_video);

	return tracker_options;
}

/**
 * Loads the sensor settings from the given config table and returns them.
 *
 * @param config the config file
 * @return the settings
 */
RgbdSensor getSensorConfig(const cpptoml::table &config)
{
	// Instantiate the RgbdSensor struct and fill it with the values from the TOML file
	refusion::RgbdSensor sensor{};
	sensor.cx = config.get_qualified_as<double>("sensor.cx").value_or(sensor.cx);
	sensor.cy = config.get_qualified_as<double>("sensor.cy").value_or(sensor.cy);
	sensor.fx = config.get_qualified_as<double>("sensor.fx").value_or(sensor.fx);
	sensor.fy = config.get_qualified_as<double>("sensor.fy").value_or(sensor.fy);
	sensor.rows = config.get_qualified_as<unsigned int>("sensor.rows").value_or(sensor.rows);
	sensor.cols = config.get_qualified_as<unsigned int>("sensor.cols").value_or(sensor.cols);
	sensor.depth_factor = config.get_qualified_as<double>("sensor.depth_factor").value_or(sensor.depth_factor);

	return sensor;
}

/**
 * Loads the settings from the given config file and returns them.
 *
 * @param config the config file
 * @return the settings
 */
settings getSettings(const cpptoml::table &config)
{
	settings s{};

	auto logging = config.get_table("logging");

    s.verbose = logging->get_qualified_as<bool>("verbose").value_or(false);
    s.debug = logging->get_qualified_as<bool>("debug").value_or(false);
    s.writeLogsToFile = logging->get_qualified_as<bool>("writeLogsToFile").value_or(false);
    s.filepath = logging->get_qualified_as<std::string>("filepath").value_or("");

    auto input = config.get_table("input");
    s.streamFrames = input->get_qualified_as<bool>("streamFrames").value_or(false);

    auto output = config.get_table("output");
    s.outputMesh = output->get_qualified_as<bool>("outputMesh").value_or(false);
    s.outputResults = output->get_qualified_as<bool>("outputResults").value_or(false);
    s.outputTimings = output->get_qualified_as<bool>("outputTimings").value_or(false);
	s.outputReprojectedVideo = output->get_qualified_as<bool>("outputReprojectedVideo").value_or(false);
	return s;
}

#endif //REFUSION_SETTINGS_H
