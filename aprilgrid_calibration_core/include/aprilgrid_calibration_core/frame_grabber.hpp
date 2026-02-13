#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include <deque>
#include <vector>
#include <memory>

#include "aprilgrid_detector_interfaces/msg/april_tag_array.hpp"

using aprilgrid_detector_interfaces::msg::AprilTagArray;

// ------------------ Structures ------------------

struct FrameDetections
{
    rclcpp::Time stamp;
    std::vector<int> ids;            // global corner ids
    std::vector<cv::Point2f> pixels;
};

struct StereoFrame
{
    FrameDetections cam0;
    FrameDetections cam1;
};

struct AprilGridGeometry
{
    int rows = 7;
    int cols = 10;
    double tag_size = 0.05;
    double spacing  = 0.015;
    int first_id = 0;

    cv::Point3f cornerPoint(int tag_id, int corner_id) const;
};

// ------------------ FrameGrabber Class ------------------

class FrameGrabber
{
public:
    FrameGrabber(const rclcpp::Logger& logger) : logger_(logger) {}
    
    // Callbacks
    void addDetectionCam0(const FrameDetections f);
    void addDetectionCam1(const FrameDetections f);

    // Access collected frames
    const std::vector<FrameDetections>& getMonoFramesCam0() const { return mono_cam0_frames_; }
    const std::vector<FrameDetections>& getMonoFramesCam1() const { return mono_cam1_frames_; }
    const std::vector<StereoFrame>& getStereoFrames() const { return stereo_frames_; }

    size_t getMonoFrameCount(int cam_id) const { return (cam_id == 0) ? mono_cam0_frames_.size() : mono_cam1_frames_.size(); }
    size_t getStereoFrameCount() const {return stereo_frames_.size(); };

    // Get 2D/3D pairs for calibration
    bool getMonoDetections(int cam_id, std::vector<std::vector<cv::Point2f>>& imagePoints, std::vector<std::vector<cv::Point3f>>& objectPoints) const;
    bool getStereoDetections(std::vector<std::vector<cv::Point2f>>& imgPts0, std::vector<std::vector<cv::Point2f>>& imgPts1, std::vector<std::vector<cv::Point3f>>& objPts) const;

    // Parameters
    void setImageSize(int width, int height) { image_width_ = width; image_height_ = height; }
    void setGrid(const AprilGridGeometry& grid) { grid_ = grid; }
    
private:

    rclcpp::Logger logger_;
    // ---------------- Frame processing ----------------
    bool goodFrame(const FrameDetections& f, const std::vector<FrameDetections>& mono_cam_frames);
    bool isDifferentEnough(const FrameDetections& last, const FrameDetections& current) const;
    bool isDiverseEnough(const FrameDetections& f, const std::vector<FrameDetections>& mono_cam_frames_) const;
    void tryBuildStereo(const FrameDetections& f, std::deque<FrameDetections>& other_buffer);
    StereoFrame buildStereo(const FrameDetections& f0, const FrameDetections& f1);

    // ---------------- Buffers ----------------
    std::deque<FrameDetections> cam0_buffer_;
    std::deque<FrameDetections> cam1_buffer_;

    std::vector<FrameDetections> mono_cam0_frames_;
    std::vector<FrameDetections> mono_cam1_frames_;
    std::vector<StereoFrame> stereo_frames_;

 
    // ---------------- Parameters ----------------
    int image_width_  = 1280;
    int image_height_ = 1024;
    AprilGridGeometry grid_;

    const float min_move_pixels_ = 15.0f;
    const int min_detections_ = 20;
    const double max_dt_ = 0.1; // 100ms max pour stéréo
};
