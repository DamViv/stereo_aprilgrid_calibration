#include "aprilgrid_calibration_core/frame_collector.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace aprilgrid_calibration_core
{

// ============================================================================
// AprilGridGeometry Implementation
// ============================================================================

cv::Point3f AprilGridGeometry::cornerPoint(int tag_id, int corner_id) const
{
    int local_id = tag_id - first_id;
    int row = local_id / cols;
    int col = local_id % cols;

    double X0 = col * (tag_size + spacing);
    double Y0 = row * (tag_size + spacing);

    switch (corner_id)
    {
        case 0: return cv::Point3f(X0,             Y0,              0);
        case 1: return cv::Point3f(X0 + tag_size,  Y0,              0);
        case 2: return cv::Point3f(X0 + tag_size,  Y0 + tag_size,   0);
        case 3: return cv::Point3f(X0,             Y0 + tag_size,   0);
        default: return cv::Point3f(0, 0, 0);
    }
}

// ============================================================================
// FrameCollector Implementation
// ============================================================================

FrameCollector::FrameCollector(const rclcpp::Logger& logger)
    : logger_(logger)
{
}

void FrameCollector::setImageSize(int width, int height)
{
    image_width_ = width;
    image_height_ = height;
}

void FrameCollector::setGrid(const AprilGridGeometry& grid)
{
    grid_ = grid;
}

void FrameCollector::setCollectionMode(CollectionMode mode)
{
    mode_ = mode;
    RCLCPP_INFO(logger_, "Collection mode changed to: %d", static_cast<int>(mode));
}

void FrameCollector::setMinFrames(int mono, int stereo)
{
    min_mono_frames_ = mono;
    min_stereo_frames_ = stereo;
}

void FrameCollector::setQualityThresholds(float min_pattern_size_ratio, float min_movement, double max_time_delta)
{
    min_pattern_size_ratio_ = min_pattern_size_ratio;
    min_move_pixels_ = min_movement;
    max_time_delta_ = max_time_delta;
}

// ============================================================================
// Frame Addition
// ============================================================================

void FrameCollector::addDetectionCam0(const FrameDetections& f)
{
    if (mode_ == CollectionMode::IDLE) {
        return;
    }

    if (mode_ != CollectionMode::MONO_CAM0 && 
        mode_ != CollectionMode::STEREO) {
        return;
    }

    if (isGoodFrame(f, mono_cam0_frames_))
    {

        mono_cam0_frames_.push_back(f);
        cam0_buffer_.push_back(f);
        // RCLCPP_INFO(logger_, "Cam0 mono frames: %zu", mono_cam0_frames_.size());
        
        // Try to build stereo if in stereo mode
        if (mode_ == CollectionMode::STEREO) {
            tryBuildStereo(f, cam1_buffer_);
        }
    }
}

void FrameCollector::addDetectionCam1(const FrameDetections& f)
{
    if (mode_ == CollectionMode::IDLE) {
        return;
    }

    if (mode_ != CollectionMode::MONO_CAM1 && 
        mode_ != CollectionMode::STEREO) {
        return;
    }

    if (isGoodFrame(f, mono_cam1_frames_))
    {

        mono_cam1_frames_.push_back(f);
        cam1_buffer_.push_back(f);
        // RCLCPP_INFO(logger_, "Cam1 mono frames: %zu", mono_cam1_frames_.size());

        // Try to build stereo if in stereo mode
        if (mode_ == CollectionMode::STEREO) {
            tryBuildStereo(f, cam0_buffer_);
        }
    }
}

// ============================================================================
// Quality Assessment
// ============================================================================

bool FrameCollector::isGoodFrame(const FrameDetections& f, 
                                 const std::vector<FrameDetections>& existing_frames) const
{
    // Check minimum number of detections
    if (f.pixels.size() < static_cast<size_t>(min_detections_)) {
        RCLCPP_DEBUG(logger_, "Frame rejected: too few detections (%zu < %d)", 
                    f.pixels.size(), min_detections_);
        return false;
    }

    // Check coverage
    float coverage = computeCoverage(f);
    if (coverage < min_pattern_size_ratio_) {
        RCLCPP_DEBUG(logger_, "Frame rejected: insufficient coverage (%.2f < %.2f)", 
                    coverage, min_pattern_size_ratio_);
        return false;
    }

    // Check diversity against existing frames
    if (!existing_frames.empty()) {
        float diversity = computeDiversity(f, existing_frames);
        if (diversity < min_diversity_) {
            RCLCPP_DEBUG(logger_, "Frame rejected: too similar to existing frames (diversity: %.2f)", 
                        diversity);
            return false;
        }
    }

    return true;
}

float FrameCollector::computeCoverage(const FrameDetections& f) const
{
    if (f.pixels.empty()) {
        return 0.0f;
    }

    // Compute bounding box of detected corners
    float xmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymin = std::numeric_limits<float>::max();
    float ymax = std::numeric_limits<float>::lowest();

    for (const auto& p : f.pixels) {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }

    // Compute coverage ratio
    float bbox_width = xmax - xmin;
    float bbox_height = ymax - ymin;
    
    float coverage_x = bbox_width / static_cast<float>(image_width_);
    float coverage_y = bbox_height / static_cast<float>(image_height_);
    
    // Return average coverage in both dimensions
    return (coverage_x + coverage_y) / 2.0f;
}

float FrameCollector::computeDiversity(const FrameDetections& f,
                                       const std::vector<FrameDetections>& existing_frames) const
{
    if (existing_frames.empty()) {
        return 1.0f;  // First frame is always diverse
    }

    // Find minimum distance to any existing frame
    float min_distance = std::numeric_limits<float>::max();

    for (const auto& existing : existing_frames) {
        float distance = 0.0f;
        int common_points = 0;

        // Compare positions of common corners
        for (size_t i = 0; i < f.ids.size(); ++i) {
            auto it = std::find(existing.ids.begin(), existing.ids.end(), f.ids[i]);
            if (it != existing.ids.end()) {
                size_t j = std::distance(existing.ids.begin(), it);
                float dx = f.pixels[i].x - existing.pixels[j].x;
                float dy = f.pixels[i].y - existing.pixels[j].y;
                distance += std::sqrt(dx*dx + dy*dy);
                common_points++;
            }
        }

        if (common_points > 0) {
            distance /= common_points;  // Average distance
            min_distance = std::min(min_distance, distance);
        }
    }

    // Normalize by minimum movement threshold
    // Returns 1.0 if distance >= min_move_pixels_, 0.0 if distance = 0
    float diversity = std::min(1.0f, min_distance / min_move_pixels_);
    return diversity;
}

bool FrameCollector::hasMinMovement(const FrameDetections& current,
                                    const FrameDetections& last) const
{
    float mean_distance = 0.0f;
    int common_points = 0;

    for (size_t i = 0; i < current.ids.size(); ++i) {
        auto it = std::find(last.ids.begin(), last.ids.end(), current.ids[i]);
        if (it != last.ids.end()) {
            size_t j = std::distance(last.ids.begin(), it);
            float dx = current.pixels[i].x - last.pixels[j].x;
            float dy = current.pixels[i].y - last.pixels[j].y;
            mean_distance += std::sqrt(dx*dx + dy*dy);
            common_points++;
        }
    }

    if (common_points == 0) {
        return true;  // No common points, consider as moved
    }

    mean_distance /= common_points;
    return mean_distance >= min_move_pixels_;
}

// ============================================================================
// Stereo Matching
// ============================================================================

void FrameCollector::tryBuildStereo(const FrameDetections& f, 
                                    std::deque<FrameDetections>& other_buffer)
{
    if (other_buffer.empty()) {
        return;
    }

    // Find nearest frame in time
    const FrameDetections* nearest = nullptr;
    double best_dt = std::numeric_limits<double>::max();

    for (const auto& bf : other_buffer) {
        double dt = std::fabs((bf.stamp - f.stamp).seconds());
        if (dt < best_dt) {
            best_dt = dt;
            nearest = &bf;
        }
    }

    // Build stereo pair if within time threshold
    if (nearest && best_dt < max_time_delta_) {
        StereoFrame sf = buildStereoFrame(f, *nearest, best_dt);
        
        // Only add if we have enough common points
        if (sf.cam0.pixels.size() >= 10) {
            stereo_frames_.push_back(sf);
            RCLCPP_INFO(logger_, "Stereo frames: %zu, dt=%.4f sec, common points: %zu", 
                       stereo_frames_.size(), best_dt, sf.cam0.pixels.size());
        }
    }

    // Clean old frames from buffer
    cleanOldBuffers(f.stamp);
}

StereoFrame FrameCollector::buildStereoFrame(const FrameDetections& f0,
                                             const FrameDetections& f1,
                                             double time_delta) const
{
    StereoFrame sf;
    sf.time_delta = time_delta;

    // Find common corner IDs
    for (size_t i = 0; i < f0.ids.size(); ++i) {
        int id = f0.ids[i];
        auto it = std::find(f1.ids.begin(), f1.ids.end(), id);
        
        if (it != f1.ids.end()) {
            size_t j = std::distance(f1.ids.begin(), it);
            
            sf.cam0.ids.push_back(id);
            sf.cam0.pixels.push_back(f0.pixels[i]);
            
            sf.cam1.ids.push_back(id);
            sf.cam1.pixels.push_back(f1.pixels[j]);
        }
    }

    sf.cam0.stamp = f0.stamp;
    sf.cam1.stamp = f1.stamp;

    return sf;
}

void FrameCollector::cleanOldBuffers(const rclcpp::Time& current_time)
{
    // Remove frames older than 2*max_time_delta from buffers
    double max_age = 2.0 * max_time_delta_;

    while (!cam0_buffer_.empty() && 
           (current_time - cam0_buffer_.front().stamp).seconds() > max_age) {
        cam0_buffer_.pop_front();
    }

    while (!cam1_buffer_.empty() && 
           (current_time - cam1_buffer_.front().stamp).seconds() > max_age) {
        cam1_buffer_.pop_front();
    }
}

// ============================================================================
// Data Extraction for Calibration
// ============================================================================

bool FrameCollector::getMonoDetections(
    int cam_id,
    std::vector<std::vector<cv::Point2f>>& imagePoints,
    std::vector<std::vector<cv::Point3f>>& objectPoints) const
{
    imagePoints.clear();
    objectPoints.clear();

    const auto& frames = (cam_id == 0) ? mono_cam0_frames_ : mono_cam1_frames_;

    if (frames.empty()) {
        return false;
    }

    for (const auto& f : frames) {
        std::vector<cv::Point2f> img;
        std::vector<cv::Point3f> obj;

        for (size_t i = 0; i < f.ids.size(); ++i) {
            int global_id = f.ids[i];
            int tag_id = global_id / 4;
            int corner_id = global_id % 4;

            img.push_back(f.pixels[i]);
            obj.push_back(grid_.cornerPoint(tag_id, corner_id));
        }

        if (!img.empty()) {
            imagePoints.push_back(img);
            objectPoints.push_back(obj);
        }
    }

    // Minimum threshold for stable calibration
    return imagePoints.size() >= 5;
}

bool FrameCollector::getStereoDetections(
    std::vector<std::vector<cv::Point2f>>& imgPts0,
    std::vector<std::vector<cv::Point2f>>& imgPts1,
    std::vector<std::vector<cv::Point3f>>& objPts) const
{
    imgPts0.clear();
    imgPts1.clear();
    objPts.clear();

    if (stereo_frames_.empty()) {
        return false;
    }

    for (const auto& sf : stereo_frames_) {
        const auto& f0 = sf.cam0;
        const auto& f1 = sf.cam1;

        // Safety check
        if (f0.ids.size() != f1.ids.size()) {
            continue;
        }

        std::vector<cv::Point2f> img0;
        std::vector<cv::Point2f> img1;
        std::vector<cv::Point3f> obj;

        for (size_t i = 0; i < f0.ids.size(); ++i) {
            int global_id = f0.ids[i];

            // Additional safety check
            if (f1.ids[i] != global_id) {
                continue;
            }

            int tag_id = global_id / 4;
            int corner_id = global_id % 4;

            img0.push_back(f0.pixels[i]);
            img1.push_back(f1.pixels[i]);
            obj.push_back(grid_.cornerPoint(tag_id, corner_id));
        }

        // Keep only pairs with enough common points
        if (img0.size() >= 10) {
            imgPts0.push_back(img0);
            imgPts1.push_back(img1);
            objPts.push_back(obj);
        }
    }

    // Minimum threshold for stable stereo calibration
    return imgPts0.size() >= 5;
}

// ============================================================================
// Statistics
// ============================================================================

CollectionStats FrameCollector::getStats() const
{
    CollectionStats stats;
    
    stats.cam0_count = static_cast<uint32_t>(mono_cam0_frames_.size());
    stats.cam1_count = static_cast<uint32_t>(mono_cam1_frames_.size());
    stats.stereo_count = static_cast<uint32_t>(stereo_frames_.size());
    
    // Compute average coverage
    if (!mono_cam0_frames_.empty()) {
        float total_coverage = 0.0f;
        for (const auto& f : mono_cam0_frames_) {
            total_coverage += computeCoverage(f);
        }
        stats.cam0_coverage = total_coverage / mono_cam0_frames_.size();
    }
    
    if (!mono_cam1_frames_.empty()) {
        float total_coverage = 0.0f;
        for (const auto& f : mono_cam1_frames_) {
            total_coverage += computeCoverage(f);
        }
        stats.cam1_coverage = total_coverage / mono_cam1_frames_.size();
    }
    
    // Compute average diversity (simplified: just check if frames are well distributed)
    stats.cam0_diversity = std::min(1.0f, stats.cam0_count / 40.0f);
    stats.cam1_diversity = std::min(1.0f, stats.cam1_count / 40.0f);
    
    // Readiness flags
    stats.cam0_ready = (stats.cam0_count >= static_cast<uint32_t>(min_mono_frames_));
    stats.cam1_ready = (stats.cam1_count >= static_cast<uint32_t>(min_mono_frames_));
    stats.stereo_ready = (stats.stereo_count >= static_cast<uint32_t>(min_stereo_frames_)) && 
                         stats.cam0_ready && stats.cam1_ready;
    

    
    return stats;
}

// ============================================================================
// Management
// ============================================================================

void FrameCollector::clearFrames(int target)
{
    switch (target) {
        case 0: // Clear all
            mono_cam0_frames_.clear();
            mono_cam1_frames_.clear();
            stereo_frames_.clear();
            cam0_buffer_.clear();
            cam1_buffer_.clear();
            RCLCPP_INFO(logger_, "Cleared all frames");
            break;
            
        case 1: // Clear cam0
            mono_cam0_frames_.clear();
            cam0_buffer_.clear();
            RCLCPP_INFO(logger_, "Cleared cam0 frames");
            break;
            
        case 2: // Clear cam1
            mono_cam1_frames_.clear();
            cam1_buffer_.clear();
            RCLCPP_INFO(logger_, "Cleared cam1 frames");
            break;
            
        case 3: // Clear stereo
            stereo_frames_.clear();
            RCLCPP_INFO(logger_, "Cleared stereo frames");
            break;
            
        default:
            RCLCPP_WARN(logger_, "Unknown clear target: %d", target);
            break;
    }
}

void FrameCollector::reset()
{
    clearFrames(0);
    mode_ = CollectionMode::IDLE;
    RCLCPP_INFO(logger_, "FrameCollector reset");
}

} // namespace aprilgrid_calibration_core
