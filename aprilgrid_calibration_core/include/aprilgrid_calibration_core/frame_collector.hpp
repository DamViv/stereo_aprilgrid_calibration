#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include <deque>
#include <vector>
#include <memory>

#include "aprilgrid_detector_interfaces/msg/april_tag_array.hpp"

using aprilgrid_detector_interfaces::msg::AprilTagArray;

namespace aprilgrid_calibration_core
{

// ------------------ Structures ------------------

struct FrameDetections
{
    rclcpp::Time stamp;
    std::vector<int> ids;            // global corner ids
    std::vector<cv::Point2f> pixels;
    
    // Quality metrics
    float coverage = 0.0f;           // Image area coverage [0-1]
    float diversity = 0.0f;          // Pose diversity relative to existing frames
};

struct StereoFrame
{
    FrameDetections cam0;
    FrameDetections cam1;
    double time_delta;               // Time difference between frames (seconds)
};

struct AprilGridGeometry
{
    int rows = 7;
    int cols = 10;
    double tag_size = 0.05;
    double spacing  = 0.015;
    int first_id = 0;

    cv::Point3f cornerPoint(int tag_id, int corner_id) const;
    int totalTags() const { return rows * cols; }
    int totalCorners() const { return rows * cols * 4; }
};

struct CollectionStats
{
    uint32_t cam0_count = 0;
    uint32_t cam1_count = 0;
    uint32_t stereo_count = 0;
    
    float cam0_coverage = 0.0f;
    float cam1_coverage = 0.0f;
    float cam0_diversity = 0.0f;
    float cam1_diversity = 0.0f;
    
    bool cam0_ready = false;
    bool cam1_ready = false;
    bool stereo_ready = false;
};

// ------------------ Collection Modes ------------------

enum class CollectionMode
{
    IDLE,
    MONO_CAM0,
    MONO_CAM1,
    STEREO,
};

// ------------------ FrameCollector Class ------------------

/**
 * @brief Collects and filters calibration frames from AprilGrid detections
 * 
 * Features:
 * - Smart frame filtering (movement, coverage, diversity)
 * - Stereo pair matching with timestamp synchronization
 * - Quality metrics computation
 * - Configurable collection modes
 */
class FrameCollector
{
public:
    FrameCollector(const rclcpp::Logger& logger);
    
    // Configuration
    void setImageSize(int width, int height);
    void setGrid(const AprilGridGeometry& grid);
    void setCollectionMode(CollectionMode mode);
    void setMinFrames(int mono, int stereo);
    void setQualityThresholds(float min_coverage, float min_movement, double max_time_delta);
    
    // Frame processing
    void addDetectionCam0(const FrameDetections& f);
    void addDetectionCam1(const FrameDetections& f);
    
    // Data access
    const std::vector<FrameDetections>& getMonoFramesCam0() const { return mono_cam0_frames_; }
    const std::vector<FrameDetections>& getMonoFramesCam1() const { return mono_cam1_frames_; }
    const std::vector<StereoFrame>& getStereoFrames() const { return stereo_frames_; }
    
    CollectionStats getStats() const;
    
    // Calibration data extraction
    bool getMonoDetections(int cam_id, 
                          std::vector<std::vector<cv::Point2f>>& imagePoints,
                          std::vector<std::vector<cv::Point3f>>& objectPoints) const;
    
    bool getStereoDetections(std::vector<std::vector<cv::Point2f>>& imgPts0,
                            std::vector<std::vector<cv::Point2f>>& imgPts1,
                            std::vector<std::vector<cv::Point3f>>& objPts) const;
    
    // Management
    void clearFrames(int target = 0); // 0=all, 1=cam0, 2=cam1, 3=stereo
    void reset();
    
private:
    rclcpp::Logger logger_;
    
    // Configuration
    int image_width_  = 1280;
    int image_height_ = 1024;
    AprilGridGeometry grid_;
    CollectionMode mode_ = CollectionMode::IDLE;
    
    // Quality thresholds
    int min_mono_frames_ = 40;
    int min_stereo_frames_ = 30;
    float min_pattern_size_ratio_ = 0.15f;          // Minimum image coverage
    float min_move_pixels_ = 20.0f;       // Minimum movement between frames
    float min_diversity_ = 0.1f;          // Minimum diversity score
    double max_time_delta_ = 0.1;         // Max time diff for stereo pairs (seconds)
    int min_detections_ = 20;             // Minimum detected corners
    
    // Buffers
    std::deque<FrameDetections> cam0_buffer_;
    std::deque<FrameDetections> cam1_buffer_;
    
    // Collected frames
    std::vector<FrameDetections> mono_cam0_frames_;
    std::vector<FrameDetections> mono_cam1_frames_;
    std::vector<StereoFrame> stereo_frames_;
    
    // Quality assessment
    bool isGoodFrame(const FrameDetections& f, 
                    const std::vector<FrameDetections>& existing_frames) const;
    float computeCoverage(const FrameDetections& f) const;
    float computeDiversity(const FrameDetections& f,
                          const std::vector<FrameDetections>& existing_frames) const;
    bool hasMinMovement(const FrameDetections& current,
                       const FrameDetections& last) const;
    
    // Stereo matching
    void tryBuildStereo(const FrameDetections& f, 
                       std::deque<FrameDetections>& other_buffer);
    StereoFrame buildStereoFrame(const FrameDetections& f0,
                                 const FrameDetections& f1,
                                 double time_delta) const;
    
    // Buffer management
    void cleanOldBuffers(const rclcpp::Time& current_time);
};

} // namespace aprilgrid_calibration_core
