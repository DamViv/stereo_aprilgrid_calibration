#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <yaml-cpp/yaml.h>

#include "aprilgrid_calibration_core/frame_collector.hpp"
#include "aprilgrid_calibration_core/mono_calibrator.hpp"
#include "aprilgrid_calibration_core/stereo_calibrator.hpp"
#include "aprilgrid_calibration_core/camera_model.hpp"

#include "aprilgrid_detector_interfaces/msg/april_tag_array.hpp"
#include "aprilgrid_detector_interfaces/msg/calibration_stats.hpp"
#include "aprilgrid_detector_interfaces/srv/start_collection.hpp"
#include "aprilgrid_detector_interfaces/srv/stop_collection.hpp"
#include "aprilgrid_detector_interfaces/srv/clear_frames.hpp"
#include "aprilgrid_detector_interfaces/srv/calibrate_camera.hpp"
#include "aprilgrid_detector_interfaces/srv/calibrate_stereo.hpp"

#include <filesystem>
#include <fstream>
#include <chrono>

using namespace aprilgrid_calibration_core;
using AprilTagArray = aprilgrid_detector_interfaces::msg::AprilTagArray;
using CalibrationStats = aprilgrid_detector_interfaces::msg::CalibrationStats;
using StartCollection = aprilgrid_detector_interfaces::srv::StartCollection;
using StopCollection = aprilgrid_detector_interfaces::srv::StopCollection;
using ClearFrames = aprilgrid_detector_interfaces::srv::ClearFrames;
using CalibrateCamera = aprilgrid_detector_interfaces::srv::CalibrateCamera;
using CalibrateStereo = aprilgrid_detector_interfaces::srv::CalibrateStereo;

class CalibrationManagerNode : public rclcpp::Node
{
public:
    CalibrationManagerNode() : Node("calibration_manager")
    {
        // ============================================================================
        // Parameters
        // ============================================================================
        
        int width = this->declare_parameter<int>("image_width", 1280);
        int height = this->declare_parameter<int>("image_height", 1024);
        
        // AprilGrid geometry
        AprilGridGeometry grid;
        grid.rows = this->declare_parameter<int>("aprilgrid.rows", 7);
        grid.cols = this->declare_parameter<int>("aprilgrid.cols", 10);
        grid.tag_size = this->declare_parameter<double>("aprilgrid.tag_size", 0.05);
        grid.spacing = this->declare_parameter<double>("aprilgrid.spacing", 0.015);
        grid.first_id = this->declare_parameter<int>("aprilgrid.first_id", 0);
        
        // Collection parameters
        int min_mono_frames = this->declare_parameter<int>("collection.min_mono_frames", 40);
        int min_stereo_frames = this->declare_parameter<int>("collection.min_stereo_frames", 30);
        float min_pattern_size_ratio = this->declare_parameter<float>("collection.min_pattern_size_ratio", 0.15);
        float min_movement = this->declare_parameter<float>("collection.min_movement_pixels", 25.0);
        double max_time_delta = this->declare_parameter<double>("collection.max_time_delta_stereo", 0.1);
        
        // Calibration parameters
        std::string camera_model_str = this->declare_parameter<std::string>("calibration.camera_model", "double_sphere");
        output_dir_ = this->declare_parameter<std::string>("calibration.output_dir", "~/.ros/calibrations");
        
        // Expand ~ in output directory
        if (output_dir_[0] == '~') {
            const char* home = std::getenv("HOME");
            if (home) {
                output_dir_ = std::string(home) + output_dir_.substr(1);
            }
        }
        
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_dir_);
        
        // Parse camera model
        if (camera_model_str == "double_sphere") {
            camera_model_ = CameraModel::DOUBLE_SPHERE;
        } else if (camera_model_str == "pinhole") {
            camera_model_ = CameraModel::PINHOLE;
        } else if (camera_model_str == "fisheye") {
            camera_model_ = CameraModel::FISHEYE;
        } else {
            RCLCPP_WARN(get_logger(), "Unknown camera model '%s', defaulting to double_sphere", 
                       camera_model_str.c_str());
            camera_model_ = CameraModel::DOUBLE_SPHERE;
        }
        
        // ============================================================================
        // Initialize Components
        // ============================================================================
        
        collector_ = std::make_shared<FrameCollector>(get_logger());
        collector_->setImageSize(width, height);
        collector_->setGrid(grid);
        collector_->setMinFrames(min_mono_frames, min_stereo_frames);
        collector_->setQualityThresholds(min_pattern_size_ratio, min_movement, max_time_delta);
        collector_->setCollectionMode(CollectionMode::IDLE);
        
        mono_cal0_ = std::make_shared<MonoCalibrator>(width, height, camera_model_);
        mono_cal1_ = std::make_shared<MonoCalibrator>(width, height, camera_model_);
        stereo_cal_ = std::make_shared<StereoCalibrator>(width, height, camera_model_);
        
        // ============================================================================
        // Subscriptions
        // ============================================================================
        
        auto sensor_qos = rclcpp::SensorDataQoS();
        
        std::string cam0_topic = this->declare_parameter<std::string>("topics.cam0_apriltags", "/cam_0/apriltags");
        std::string cam1_topic = this->declare_parameter<std::string>("topics.cam1_apriltags", "/cam_1/apriltags");
        
        sub_cam0_ = this->create_subscription<AprilTagArray>(
            cam0_topic, sensor_qos,
            std::bind(&CalibrationManagerNode::onCam0Detections, this, std::placeholders::_1));
        
        sub_cam1_ = this->create_subscription<AprilTagArray>(
            cam1_topic, sensor_qos,
            std::bind(&CalibrationManagerNode::onCam1Detections, this, std::placeholders::_1));
        
        // ============================================================================
        // Publishers
        // ============================================================================
        
        stats_pub_ = this->create_publisher<CalibrationStats>("/calibration/stats", 10);
        
        // Timer for publishing stats at 10Hz
        stats_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&CalibrationManagerNode::publishStats, this));
        
        // ============================================================================
        // Services
        // ============================================================================
        
        srv_start_collection_ = this->create_service<StartCollection>(
            "/calibration/start_collection",
            std::bind(&CalibrationManagerNode::onStartCollection, this, 
                     std::placeholders::_1, std::placeholders::_2));
        
        srv_stop_collection_ = this->create_service<StopCollection>(
            "/calibration/stop_collection",
            std::bind(&CalibrationManagerNode::onStopCollection, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        srv_clear_frames_ = this->create_service<ClearFrames>(
            "/calibration/clear_frames",
            std::bind(&CalibrationManagerNode::onClearFrames, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        srv_calibrate_camera_ = this->create_service<CalibrateCamera>(
            "/calibration/calibrate_camera",
            std::bind(&CalibrationManagerNode::onCalibrateCamera, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        srv_calibrate_stereo_ = this->create_service<CalibrateStereo>(
            "/calibration/calibrate_stereo",
            std::bind(&CalibrationManagerNode::onCalibrateStereo, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(get_logger(), "Calibration Manager started");
        RCLCPP_INFO(get_logger(), "  Camera model: %s", camera_model_str.c_str());
        RCLCPP_INFO(get_logger(), "  Output directory: %s", output_dir_.c_str());
        RCLCPP_INFO(get_logger(), "  AprilGrid: %dx%d tags", grid.rows, grid.cols);
    }

private:
    // ============================================================================
    // Members
    // ============================================================================
    
    std::shared_ptr<FrameCollector> collector_;
    std::shared_ptr<MonoCalibrator> mono_cal0_;
    std::shared_ptr<MonoCalibrator> mono_cal1_;
    std::shared_ptr<StereoCalibrator> stereo_cal_;
    
    CameraModel camera_model_;
    std::string output_dir_;
    
    bool cam0_calibrated_ = false;
    bool cam1_calibrated_ = false;
    
    rclcpp::Subscription<AprilTagArray>::SharedPtr sub_cam0_;
    rclcpp::Subscription<AprilTagArray>::SharedPtr sub_cam1_;
    rclcpp::Publisher<CalibrationStats>::SharedPtr stats_pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    rclcpp::Service<StartCollection>::SharedPtr srv_start_collection_;
    rclcpp::Service<StopCollection>::SharedPtr srv_stop_collection_;
    rclcpp::Service<ClearFrames>::SharedPtr srv_clear_frames_;
    rclcpp::Service<CalibrateCamera>::SharedPtr srv_calibrate_camera_;
    rclcpp::Service<CalibrateStereo>::SharedPtr srv_calibrate_stereo_;
    
    // ============================================================================
    // Detection Callbacks
    // ============================================================================
    
    void onCam0Detections(const AprilTagArray::SharedPtr msg)
    {
        FrameDetections frame = convertAprilTagArrayToFrame(msg);
        collector_->addDetectionCam0(frame);
    }
    
    void onCam1Detections(const AprilTagArray::SharedPtr msg)
    {
        FrameDetections frame = convertAprilTagArrayToFrame(msg);
        collector_->addDetectionCam1(frame);        
    }
    
    FrameDetections convertAprilTagArrayToFrame(const AprilTagArray::SharedPtr msg)
    {
        FrameDetections frame;
        frame.stamp = msg->header.stamp;
        
        for (const auto& tag : msg->tags) {
            // Filter out low-quality detections
            if (tag.hamming > 1.0) {
                continue;
            }
            
            // Each tag has 4 corners
            for (int c = 0; c < 4; ++c) {
                int global_id = tag.id * 4 + c;
                float x = tag.corners[2*c];
                float y = tag.corners[2*c + 1];
                
                frame.ids.push_back(global_id);
                frame.pixels.emplace_back(x, y);
            }
        }
        
        return frame;
    }
    
    // ============================================================================
    // Stats Publishing
    // ============================================================================
    
    void publishStats()
    {
        auto stats = collector_->getStats();
        
        auto msg = CalibrationStats();
        msg.header.stamp = this->now();
        msg.header.frame_id = "calibration";
        
        msg.cam0_frame_count = stats.cam0_count;
        msg.cam1_frame_count = stats.cam1_count;
        msg.stereo_pair_count = stats.stereo_count;
        
        msg.cam0_coverage = stats.cam0_coverage;
        msg.cam1_coverage = stats.cam1_coverage;
        msg.cam0_diversity = stats.cam0_diversity;
        msg.cam1_diversity = stats.cam1_diversity;
        
        msg.cam0_ready = stats.cam0_ready;
        msg.cam1_ready = stats.cam1_ready;
        msg.stereo_ready = stats.stereo_ready && cam0_calibrated_ && cam1_calibrated_;
        
        msg.is_collecting = (collector_->getStats().cam0_count > 0 || 
                            collector_->getStats().cam1_count > 0);
        
        // Set collection mode string
        // (Note: We'd need to add a getter for mode in FrameCollector for exact mode)
        msg.collection_mode = "ACTIVE";  // Simplified
        
        stats_pub_->publish(msg);
    }
    
    // ============================================================================
    // Service Handlers
    // ============================================================================
    
    void onStartCollection(
        const StartCollection::Request::SharedPtr req,
        StartCollection::Response::SharedPtr res)
    {
        CollectionMode mode;
        
        switch (req->mode) {
            case StartCollection::Request::MODE_IDLE:
                mode = CollectionMode::IDLE;
                break;
            case StartCollection::Request::MODE_MONO_CAM0:
                mode = CollectionMode::MONO_CAM0;
                break;
            case StartCollection::Request::MODE_MONO_CAM1:
                mode = CollectionMode::MONO_CAM1;
                break;
            case StartCollection::Request::MODE_STEREO:
                mode = CollectionMode::STEREO;
                break;
            default:
                res->success = false;
                res->message = "Invalid collection mode";
                return;
        }
        
        collector_->setCollectionMode(mode);
        
        res->success = true;
        res->message = "Collection started in mode " + std::to_string(req->mode);
        
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
    }
    
    void onStopCollection(
        const StopCollection::Request::SharedPtr,
        StopCollection::Response::SharedPtr res)
    {
        collector_->setCollectionMode(CollectionMode::IDLE);
                
        res->success = true;
        res->message = "Collection stopped";
        
        RCLCPP_INFO(get_logger(), "Collection stopped");
    }
    
    void onClearFrames(
        const ClearFrames::Request::SharedPtr req,
        ClearFrames::Response::SharedPtr res)
    {
        collector_->clearFrames(req->target);
        
        // Also reset calibration flags if clearing the corresponding camera
        if (req->target == ClearFrames::Request::CLEAR_ALL ||
            req->target == ClearFrames::Request::CLEAR_CAM0) {
            cam0_calibrated_ = false;
        }
        if (req->target == ClearFrames::Request::CLEAR_ALL ||
            req->target == ClearFrames::Request::CLEAR_CAM1) {
            cam1_calibrated_ = false;
        }
        
        res->success = true;
        res->message = "Frames cleared (target=" + std::to_string(req->target) + ")";
        
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
    }
    
    void onCalibrateCamera(
        const CalibrateCamera::Request::SharedPtr req,
        CalibrateCamera::Response::SharedPtr res)
    {
        int cam_id = req->camera_id;
        
        if (cam_id != 0 && cam_id != 1) {
            res->success = false;
            res->message = "Invalid camera_id (must be 0 or 1)";
            return;
        }
        
        RCLCPP_INFO(get_logger(), "Starting monocular calibration for camera %d...", cam_id);
        
        // Get detection data
        std::vector<std::vector<cv::Point2f>> imgPts;
        std::vector<std::vector<cv::Point3f>> objPts;
        
        if (!collector_->getMonoDetections(cam_id, imgPts, objPts)) {
            res->success = false;
            res->message = "Insufficient calibration data for camera " + std::to_string(cam_id);
            RCLCPP_ERROR(get_logger(), "%s", res->message.c_str());
            return;
        }
        
        RCLCPP_INFO(get_logger(), "  Using %zu views for calibration", imgPts.size());
        
        // Select calibrator
        auto& calibrator = (cam_id == 0) ? mono_cal0_ : mono_cal1_;
        
        // Add views
        for (size_t i = 0; i < imgPts.size(); ++i) {
            calibrator->addView(imgPts[i], objPts[i]);
        }
        
        // Calibrate
        if (!calibrator->calibrate()) {
            res->success = false;
            res->message = "Calibration failed for camera " + std::to_string(cam_id);
            RCLCPP_ERROR(get_logger(), "%s", res->message.c_str());
            return;
        }
        
        double reproj_error = calibrator->reprojectionError();
        
        // Mark as calibrated
        if (cam_id == 0) {
            cam0_calibrated_ = true;
        } else {
            cam1_calibrated_ = true;
        }
        
        // Save calibration
        std::string output_path = req->output_path.empty() ? 
            output_dir_ + "/camera_" + std::to_string(cam_id) + ".yaml" :
            req->output_path;
        
        if (saveMonoCalibration(calibrator, output_path, cam_id)) {
            res->calibration_file = output_path;
        }
        
        // Fill response
        res->success = true;
        res->message = "Camera " + std::to_string(cam_id) + " calibrated successfully";
        res->reprojection_error = reproj_error;
        
        // Extract intrinsics (simplified - would need to parse from calibrator)
        const cv::Mat& K = calibrator->cameraMatrix();
        const cv::Mat& D = calibrator->distCoeffs();
        
        res->intrinsics = {K.at<double>(0,0), K.at<double>(1,1), 
                          K.at<double>(0,2), K.at<double>(1,2)};
        
        for (int i = 0; i < D.rows; ++i) {
            res->distortion.push_back(D.at<double>(i, 0));
        }
        
        RCLCPP_INFO(get_logger(), "Camera %d calibration complete. Reprojection error: %.3f pixels",
                   cam_id, reproj_error);
        RCLCPP_INFO(get_logger(), "  Saved to: %s", output_path.c_str());
    }
    
    void onCalibrateStereo(
        const CalibrateStereo::Request::SharedPtr req,
        CalibrateStereo::Response::SharedPtr res)
    {
        if (!cam0_calibrated_ || !cam1_calibrated_) {
            res->success = false;
            res->message = "Both cameras must be calibrated before stereo calibration";
            RCLCPP_ERROR(get_logger(), "%s", res->message.c_str());
            return;
        }
        
        RCLCPP_INFO(get_logger(), "Starting stereo calibration...");
        
        // Get stereo detection data
        std::vector<std::vector<cv::Point2f>> imgPts0, imgPts1;
        std::vector<std::vector<cv::Point3f>> objPts;
        
        if (!collector_->getStereoDetections(imgPts0, imgPts1, objPts)) {
            res->success = false;
            res->message = "Insufficient stereo pairs for calibration";
            RCLCPP_ERROR(get_logger(), "%s", res->message.c_str());
            return;
        }
        
        RCLCPP_INFO(get_logger(), "  Using %zu stereo pairs for calibration", imgPts0.size());
        
        // Set intrinsics from mono calibrations
        stereo_cal_->setIntrinsics(
            mono_cal0_->cameraMatrix(), mono_cal0_->distCoeffs(),
            mono_cal1_->cameraMatrix(), mono_cal1_->distCoeffs()
        );
        
        // Add views
        for (size_t i = 0; i < imgPts0.size(); ++i) {
            stereo_cal_->addView(imgPts0[i], imgPts1[i], objPts[i]);
        }
        
        // Calibrate
        if (!stereo_cal_->calibrate()) {
            res->success = false;
            res->message = "Stereo calibration failed";
            RCLCPP_ERROR(get_logger(), "%s", res->message.c_str());
            return;
        }
        
        // --- Get stereo results ---
        cv::Mat R, t;  // Rotation & translation from stereo_cal_
        stereo_cal_->getExtrinsics(R, t);

        cv::Mat rvec;
        cv::Rodrigues(R, rvec); // Matrice R -> vecteur de rotation

        res->rotation_euler = {
            rvec.at<double>(0,0),
            rvec.at<double>(1,0),
            rvec.at<double>(2,0)
        };

        res->baseline_vector = {
            t.at<double>(0,0),
            t.at<double>(1,0),
            t.at<double>(2,0)
        };

        res->t_cam1_cam0.fill(0.0);
        for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
                res->t_cam1_cam0[i*4 + j] = R.at<double>(i,j);   // rotation
            }
            res->t_cam1_cam0[i*4 + 3] = t.at<double>(i,0);       // translation
        }
        res->t_cam1_cam0[15] = 1.0;

        res->success = true;
        res->message = "Stereo calibration completed successfully";
        res->reprojection_error = stereo_cal_->reprojectionError();  // si dispo

        // --- Save calibration ---
        std::string output_path = req->output_path.empty() ?
            output_dir_ + "/stereo_calibration.yaml" :
            req->output_path;

        if (saveStereoCalibration(stereo_cal_, output_path)) {
            res->calibration_file = output_path;
        }

        RCLCPP_INFO(get_logger(), "Stereo calibration complete. Reprojection error: %.3f", res->reprojection_error);
        RCLCPP_INFO(get_logger(), "  Saved to: %s", output_path.c_str());
    }
    
    // ============================================================================
    // Calibration Saving
    // ============================================================================
    
    bool saveMonoCalibration(const std::shared_ptr<MonoCalibrator>& cal,
                            const std::string& filepath,
                            int cam_id)
    {
        try {
            YAML::Emitter out;
            out << YAML::BeginMap;
            
            out << YAML::Key << "camera_name" << YAML::Value << ("camera_" + std::to_string(cam_id));
            out << YAML::Key << "camera_model" << YAML::Value << "double_sphere";  // TODO: Use actual model
            
            const cv::Mat& K = cal->cameraMatrix();
            const cv::Mat& D = cal->distCoeffs();
            
            out << YAML::Key << "intrinsics" << YAML::Value << YAML::BeginSeq;
            out << K.at<double>(0,0) << K.at<double>(1,1) 
                << K.at<double>(0,2) << K.at<double>(1,2);
            out << YAML::EndSeq;
            
            out << YAML::Key << "distortion_coeffs" << YAML::Value << YAML::BeginSeq;
            for (int i = 0; i < D.rows; ++i) {
                out << D.at<double>(i, 0);
            }
            out << YAML::EndSeq;
            
            out << YAML::Key << "reprojection_error" << YAML::Value << cal->reprojectionError();
            
            out << YAML::EndMap;
            
            std::ofstream file(filepath);
            file << out.c_str();
            file.close();
            
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to save calibration: %s", e.what());
            return false;
        }
    }
    
    bool saveStereoCalibration(const std::shared_ptr<StereoCalibrator>& cal,
                              const std::string& filepath)
    {
        try {
            YAML::Emitter out;
            out << YAML::BeginMap;
            
            out << YAML::Key << "stereo_calibration" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "baseline" << YAML::Value << 0.12;  // TODO: Extract actual baseline
            // TODO: Add transformation matrix, rotation, translation
            out << YAML::EndMap;
            
            out << YAML::EndMap;
            
            std::ofstream file(filepath);
            file << out.c_str();
            file.close();
            
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to save stereo calibration: %s", e.what());
            return false;
        }
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<CalibrationManagerNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("calibration_manager"), 
                    "Fatal error: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
