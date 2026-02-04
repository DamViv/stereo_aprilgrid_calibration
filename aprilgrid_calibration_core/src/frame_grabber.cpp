#include "../include/aprilgrid_calibration_core/frame_grabber.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

cv::Point3f AprilGridGeometry::cornerPoint(int tag_id, int corner_id) const
{
    int local_id = tag_id - first_id;
    int row = local_id / cols;
    int col = local_id % cols;

    double X0 = col * (tag_size + spacing);
    double Y0 = row * (tag_size + spacing);

    switch (corner_id)
    {
        case 0: return {X0,             Y0,              0};
        case 1: return {X0 + tag_size,  Y0,              0};
        case 2: return {X0 + tag_size,  Y0 + tag_size,   0};
        case 3: return {X0,             Y0 + tag_size,   0};
        default: return {0,0,0};
    }
}

// ---------------- Frame quality checks ----------------

bool FrameGrabber::goodFrame(const FrameDetections& f, const std::vector<FrameDetections>& mono_cam_frames_)
{
    if (f.pixels.size() < min_detections_) {
        RCLCPP_INFO(logger_, "Frame rejected, too few detections: %lu", f.pixels.size());
        return false;
    }

    float xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
    for (const auto& p : f.pixels)
    {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }

    if ((xmax - xmin) < 0.15 * image_width_ || 
        (ymax - ymin) < 0.15 * image_height_) {
        RCLCPP_INFO(logger_, "Frame rejected, pattern too small");
        return false;
    }

    if (!mono_cam_frames_.empty() && !isDiverseEnough(f, mono_cam_frames_)) {
        RCLCPP_INFO(logger_, "Frame rejected, too similar to previous saved views");
        return false;
    }

    return true;
}

bool FrameGrabber::isDifferentEnough(const FrameDetections& last, const FrameDetections& current) const
{
    float meanDist = 0.0f;
    int n = 0;

    for (size_t i = 0; i < current.ids.size(); ++i)
    {
        auto it = std::find(last.ids.begin(), last.ids.end(), current.ids[i]);
        if (it != last.ids.end())
        {
            size_t j = std::distance(last.ids.begin(), it);
            float dx = current.pixels[i].x - last.pixels[j].x;
            float dy = current.pixels[i].y - last.pixels[j].y;
            meanDist += std::sqrt(dx*dx + dy*dy);
            n++;
        }
    }

    if (n == 0)
        return true;

    meanDist /= n;
    return meanDist >= min_move_pixels_;
}


bool FrameGrabber::isDiverseEnough(const FrameDetections& f, const std::vector<FrameDetections>& mono_cam_frames_) const
{
    for (const auto& saved : mono_cam_frames_)
    {
        if(!isDifferentEnough(saved, f))
            return false;
    }

    return true;
}

// ---------------- Stereo builder ----------------

void FrameGrabber::tryBuildStereo(const FrameDetections& f, std::deque<FrameDetections>& other_buffer)
{
    if (other_buffer.empty()) return;

    const FrameDetections* nearest = nullptr;
    double best_dt = std::numeric_limits<double>::max();

    for (const auto& bf : other_buffer)
    {
        double dt = std::fabs((bf.stamp - f.stamp).seconds());
        if (dt < best_dt)
        {
            best_dt = dt;
            nearest = &bf;
        }
    }

    if (nearest && best_dt < max_dt_)
    {
        stereo_frames_.push_back(buildStereo(f, *nearest));
        RCLCPP_INFO(logger_, "Stereo frames: %lu, dt=%.4f", stereo_frames_.size(), best_dt);
    }

    // Remove old frames from buffer
    while (!other_buffer.empty() && (f.stamp - other_buffer.front().stamp).seconds() > 2*max_dt_)
        other_buffer.pop_front();
}

StereoFrame FrameGrabber::buildStereo(const FrameDetections& f0, const FrameDetections& f1)
{
    StereoFrame sf;

    for (size_t i = 0; i < f0.ids.size(); ++i)
    {
        int id = f0.ids[i];
        auto it = std::find(f1.ids.begin(), f1.ids.end(), id);
        if (it == f1.ids.end())
            continue;

        size_t j = std::distance(f1.ids.begin(), it);

        sf.cam0.ids.push_back(id);
        sf.cam0.pixels.push_back(f0.pixels[i]);

        sf.cam1.ids.push_back(id);
        sf.cam1.pixels.push_back(f1.pixels[j]);
    }

    sf.cam0.stamp = f0.stamp;
    sf.cam1.stamp = f1.stamp;
    return sf;
}

// ---------------- Callbacks ----------------

void FrameGrabber::addDetectionCam0(const FrameDetections f)
{    
    if (goodFrame(f, mono_cam0_frames_))
    {
        mono_cam0_frames_.push_back(f);
        cam0_buffer_.push_back(f);
        RCLCPP_INFO(logger_, "Cam0 mono frames: %lu", mono_cam0_frames_.size());

        tryBuildStereo(f, cam1_buffer_);
    }
}

void FrameGrabber::addDetectionCam1(const FrameDetections f)
{
    if (goodFrame(f, mono_cam1_frames_))
    {
        mono_cam1_frames_.push_back(f);
        cam1_buffer_.push_back(f);
        RCLCPP_INFO(logger_, "Cam1 mono frames: %lu", mono_cam1_frames_.size());

        tryBuildStereo(f, cam0_buffer_);
    }
}



// ---------------- Detection getters  ----------------

bool FrameGrabber::getMonoDetections(
    int cam_id,
    std::vector<std::vector<cv::Point2f>>& imagePoints,
    std::vector<std::vector<cv::Point3f>>& objectPoints
) const
{
    imagePoints.clear();
    objectPoints.clear();

    const auto& frames =
        (cam_id == 0) ? mono_cam0_frames_ : mono_cam1_frames_;

    if (frames.empty())
        return false;

    for (const auto& f : frames)
    {
        std::vector<cv::Point2f> img;
        std::vector<cv::Point3f> obj;

        for (size_t i = 0; i < f.ids.size(); ++i)
        {
            int global_id = f.ids[i];
            int tag_id    = global_id / 4;
            int corner_id = global_id % 4;

            img.push_back(f.pixels[i]);
            obj.push_back(grid_.cornerPoint(tag_id, corner_id));
        }

        if (!img.empty())
        {
            imagePoints.push_back(img);
            objectPoints.push_back(obj);
        }
    }

    return imagePoints.size() >= 5; // seuil minimal
}



bool FrameGrabber::getStereoDetections(
    std::vector<std::vector<cv::Point2f>>& imgPts0,
    std::vector<std::vector<cv::Point2f>>& imgPts1,
    std::vector<std::vector<cv::Point3f>>& objPts
) const
{
    imgPts0.clear();
    imgPts1.clear();
    objPts.clear();

    if (stereo_frames_.empty())
        return false;

    for (const auto& sf : stereo_frames_)
    {
        const auto& f0 = sf.cam0;
        const auto& f1 = sf.cam1;

        // Sécurité
        if (f0.ids.size() != f1.ids.size())
            continue;

        std::vector<cv::Point2f> img0;
        std::vector<cv::Point2f> img1;
        std::vector<cv::Point3f> obj;

        for (size_t i = 0; i < f0.ids.size(); ++i)
        {
            int global_id = f0.ids[i];

            // Sécurité supplémentaire
            if (f1.ids[i] != global_id)
                continue;

            int tag_id    = global_id / 4;
            int corner_id = global_id % 4;

            img0.push_back(f0.pixels[i]);
            img1.push_back(f1.pixels[i]);
            obj.push_back(grid_.cornerPoint(tag_id, corner_id));
        }

        // On garde uniquement les paires suffisamment riches
        if (img0.size() >= 10)
        {
            imgPts0.push_back(img0);
            imgPts1.push_back(img1);
            objPts.push_back(obj);
        }
    }

    // seuil minimal pour une calibration stéréo stable
    return imgPts0.size() >= 5;
}
