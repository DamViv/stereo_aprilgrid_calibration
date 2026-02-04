#include <rclcpp/rclcpp.hpp>
#include "aprilgrid_calibration_core/frame_grabber.hpp"
#include "aprilgrid_calibration_core/mono_calibrator.hpp"
#include "aprilgrid_calibration_core/stereo_calibrator.hpp"
#include "aprilgrid_calibration_core/camera_model.hpp"
#include "aprilgrid_calibration_core/projections_double_sphere.hpp"
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>

class AprilGridCalibrationNode : public rclcpp::Node
{
public:
    AprilGridCalibrationNode() : Node("aprilgrid_calibration")
    {
        // ---------------- Parameters ----------------
        int width  = this->declare_parameter<int>("image_width", 1280);
        int height = this->declare_parameter<int>("image_height", 1024);
        
        frame_grabber_ = std::make_shared<FrameGrabber>(this->get_logger());        
        frame_grabber_->setImageSize(width, height);
        
        mono_calibrator0_ = std::make_shared<MonoCalibrator>(width, height, CameraModel::DOUBLE_SPHERE);
        mono_calibrator1_ = std::make_shared<MonoCalibrator>(width, height, CameraModel::DOUBLE_SPHERE);
        stereo_calibrator_ = std::make_shared<StereoCalibrator>(width, height, CameraModel::DOUBLE_SPHERE);

        AprilGridGeometry grid;
        grid.rows     = this->declare_parameter<int>("aprilgrid.rows", 7);
        grid.cols     = this->declare_parameter<int>("aprilgrid.cols", 10);
        grid.tag_size = this->declare_parameter<double>("aprilgrid.tag_size", 0.05);
        grid.spacing  = this->declare_parameter<double>("aprilgrid.spacing", 0.015);
        grid.first_id = this->declare_parameter<int>("aprilgrid.first_id", 0);
        frame_grabber_->setGrid(grid);

        // ---------------- Subscriptions ----------------
        auto sensor_qos = rclcpp::SensorDataQoS();

        sub_cam0_ = this->create_subscription<AprilTagArray>(
          "/cam_0/apriltags", sensor_qos,
          std::bind(&AprilGridCalibrationNode::cbCam0, this, std::placeholders::_1));

        sub_cam1_ = this->create_subscription<AprilTagArray>(
          "/cam_1/apriltags", sensor_qos,
          std::bind(&AprilGridCalibrationNode::cbCam1, this, std::placeholders::_1));        

        RCLCPP_INFO(get_logger(), "AprilGrid calibration node started");

        // cam_sub = this->create_subscription<sensor_msgs::msg::Image>(
        //         "/cam_0/image_raw", sensor_qos,
        //         std::bind(&AprilGridCalibrationNode::cbImg0, this, std::placeholders::_1)); 
    }

private:
    std::shared_ptr<FrameGrabber> frame_grabber_;
    std::shared_ptr<MonoCalibrator> mono_calibrator0_;
    std::shared_ptr<MonoCalibrator> mono_calibrator1_;
    std::shared_ptr<StereoCalibrator> stereo_calibrator_;

    rclcpp::Subscription<AprilTagArray>::SharedPtr sub_cam0_;
    rclcpp::Subscription<AprilTagArray>::SharedPtr sub_cam1_;
    // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam_sub;


    // void cbImg0(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    // {
    //     // if(mono_calibrator_->reprojectionError() != -1.0)
    //     // {
    //     //   cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
    //     //   cv::Mat equi = generateEquirectangular(img, mono_calibrator_->cameraMatrix(), mono_calibrator_->distCoeffs(), 1000, 1000);
          
    //     //   cv::Mat undis = undistortDoubleSphere(img, mono_calibrator_->cameraMatrix(), mono_calibrator_->distCoeffs());
    //     //   cv::imshow("Undistorted", undis);
    //     //   cv::imshow("Equirectangular", equi);
    //     //   cv::waitKey(1);
    //     // }        
    // }

    void cbCam0(const AprilTagArray::SharedPtr msg)
    {
        FrameDetections f0 = getDetectionsFromMsg(msg);
        frame_grabber_->addDetectionCam0(f0);   
        
        if(frame_grabber_->getMonoFrameCount(0) >= 40 && !mono_calibrator0_->ready())
        {
            std::vector<std::vector<cv::Point2f>> imgPts;
            std::vector<std::vector<cv::Point3f>> objPts;

            if (frame_grabber_->getMonoDetections(0, imgPts, objPts))
            {
                for (size_t i = 0; i < imgPts.size(); ++i)
                {
                    mono_calibrator0_->addView(imgPts[i], objPts[i]);
                }

                if (mono_calibrator0_->calibrate())
                {
                    RCLCPP_INFO(get_logger(), "Mono calibration done for cam0. Reprojection error: %.3f", mono_calibrator0_->reprojectionError());
                }
            }
        }

        if(frame_grabber_->getStereoFrameCount() >= 30 && !stereo_calibrator_->ready())
        {
            std::vector<std::vector<cv::Point2f>> imgPts0, imgPts1;
            std::vector<std::vector<cv::Point3f>> objPts;

            if (frame_grabber_->getStereoDetections(imgPts0, imgPts1, objPts))
            {
                for (size_t i = 0; i < imgPts0.size(); ++i)
                {
                    stereo_calibrator_->addView(imgPts0[i], imgPts1[i], objPts[i]);
                }

                stereo_calibrator_->setIntrinsics(
                    mono_calibrator0_->cameraMatrix(),
                    mono_calibrator0_->distCoeffs(),
                    mono_calibrator1_->cameraMatrix(),
                    mono_calibrator1_->distCoeffs()
                );
                if (stereo_calibrator_->calibrate())
                {
                    RCLCPP_INFO(get_logger(), "Stereo calibration done.");
                }
            }
        }
    };

    void cbCam1(const AprilTagArray::SharedPtr msg)
    {
        FrameDetections f1 = getDetectionsFromMsg(msg);
        frame_grabber_->addDetectionCam1(f1);        

        if(frame_grabber_->getMonoFrameCount(1) >= 40 && !mono_calibrator1_->ready())
        {
            std::vector<std::vector<cv::Point2f>> imgPts;
            std::vector<std::vector<cv::Point3f>> objPts;

            if (frame_grabber_->getMonoDetections(1, imgPts, objPts))
            {
                for (size_t i = 0; i < imgPts.size(); ++i)
                {
                    mono_calibrator1_->addView(imgPts[i], objPts[i]);
                }

                if (mono_calibrator1_->calibrate())
                {
                    RCLCPP_INFO(get_logger(), "Mono calibration done for cam1. Reprojection error: %.3f", mono_calibrator1_->reprojectionError());
                }
            }
        }

    };

    FrameDetections getDetectionsFromMsg(const AprilTagArray::SharedPtr msg)
    {
        FrameDetections f;
        f.stamp = msg->header.stamp;

        for (const auto& tag : msg->tags)
        {
            if (tag.hamming > 1.0)
                continue;

            for (int c = 0; c < 4; ++c)
            {
                int global_id = tag.id * 4 + c;
                float x = tag.corners[2*c];
                float y = tag.corners[2*c + 1];

                f.ids.push_back(global_id);
                f.pixels.emplace_back(x, y);
            }
        }
        return f;
    }
  };

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AprilGridCalibrationNode>());
    rclcpp::shutdown();
    return 0;
}




// // aprilgrid_calibration_node.cpp
// // ROS 2 Jazzy - AprilGrid calibration collector (mono + stereo)

// #include <rclcpp/rclcpp.hpp>
// #include <opencv2/core.hpp>
// #include <optional>
// #include <vector>
// #include <cmath>
// #include <deque>


// #include "aprilgrid_detector_interfaces/msg/april_tag.hpp"
// #include "aprilgrid_detector_interfaces/msg/april_tag_array.hpp"

// using aprilgrid_detector_interfaces::msg::AprilTagArray;

// // ------------------ Structures ------------------

// struct FrameDetections
// {
//   rclcpp::Time stamp;
//   std::vector<int> ids;            // global corner ids
//   std::vector<cv::Point2f> pixels;
// };

// struct StereoFrame
// {
//   FrameDetections cam0;
//   FrameDetections cam1;
// };

// struct AprilGridGeometry
// {
//   int rows = 10;
//   int cols = 7;
//   double tag_size = 0.05;
//   double spacing  = 0.015;
//   int first_id = 0;

//   cv::Point3f cornerPoint(int tag_id, int corner_id) const
//   {
//     int local_id = tag_id - first_id;
//     int row = local_id / cols;
//     int col = local_id % cols;

//     double X0 = col * (tag_size + spacing);
//     double Y0 = row * (tag_size + spacing);

//     switch (corner_id)
//     {
//       case 0: return {X0,             Y0,              0};
//       case 1: return {X0 + tag_size,  Y0,              0};
//       case 2: return {X0 + tag_size,  Y0 + tag_size,   0};
//       case 3: return {X0,             Y0 + tag_size,   0};
//       default: return {0,0,0};
//     }
//   }
// };

// // ------------------ Main Node ------------------

// class AprilGridCalibrationNode : public rclcpp::Node
// {
// public:
//   AprilGridCalibrationNode() : Node("aprilgrid_calibration")
//   {
//     // parameters
//     image_width_  = this->declare_parameter<int>("image_width", 1280);
//     image_height_ = this->declare_parameter<int>("image_height", 1024);

//     grid_.rows      = this->declare_parameter<int>("aprilgrid.rows", 10);
//     grid_.cols      = this->declare_parameter<int>("aprilgrid.cols", 7);
//     grid_.tag_size  = this->declare_parameter<double>("aprilgrid.tag_size", 0.05);
//     grid_.spacing   = this->declare_parameter<double>("aprilgrid.spacing", 0.015);
//     grid_.first_id  = this->declare_parameter<int>("aprilgrid.first_id", 0);

//     // AprilTag detections msgs subscribers
//     auto sensor_qos = rclcpp::SensorDataQoS();

//     sub_cam0_ = this->create_subscription<AprilTagArray>(
//       "/cam_0/apriltags", sensor_qos,
//       std::bind(&AprilGridCalibrationNode::cbCam0, this, std::placeholders::_1));

//     sub_cam1_ = this->create_subscription<AprilTagArray>(
//       "/cam_1/apriltags", sensor_qos,
//       std::bind(&AprilGridCalibrationNode::cbCam1, this, std::placeholders::_1));

//     RCLCPP_INFO(get_logger(), "AprilGrid calibration node started");
//   }

// private:
//   enum Mode { IDLE, COLLECT_CAM0, COLLECT_CAM1, COLLECT_STEREO };
//   Mode mode_ = COLLECT_CAM0;   // par défaut on collecte stéréo

//   // buffers
//   std::deque<FrameDetections> cam0_buffer_;
//   std::deque<FrameDetections> cam1_buffer_;

//   // saved frame for calibration
//   std::vector<FrameDetections> mono_cam0_frames_;
//   std::vector<FrameDetections> mono_cam1_frames_;
//   std::vector<StereoFrame> stereo_frames_;
  
//   std::shared_ptr<FrameDetections> last_cam0_;
//   std::shared_ptr<FrameDetections> last_cam1_;

//   int image_width_;
//   int image_height_;
//   AprilGridGeometry grid_;

//   rclcpp::Subscription<AprilTagArray>::SharedPtr sub_cam0_;
//   rclcpp::Subscription<AprilTagArray>::SharedPtr sub_cam1_;

//   const float min_move_pixels_ = 20.0f;
//   const int min_detections_ = 20;
//   const double max_dt_ = 0.1;


//   // ---------------- Parsing ----------------

//   FrameDetections getDetectionsFromMsg(const AprilTagArray::SharedPtr msg)
//   {
//     FrameDetections f;
//     f.stamp = msg->header.stamp;

//     for (const auto& tag : msg->tags)
//     {
//       if (tag.hamming > 1.0)
//       {
//         RCLCPP_INFO(get_logger(), "Tag rejected: %f", tag.hamming);
//         continue;
//       }        

//       for (int c = 0; c < 4; ++c)
//       {
//         int global_id = tag.id * 4 + c;
//         float x = tag.corners[2*c];
//         float y = tag.corners[2*c + 1];

//         f.ids.push_back(global_id);
//         f.pixels.emplace_back(x, y);
//       }
//     }

//     return f;
//   }

//   // ---------------- Frame quality checks ----------------

//   bool goodFrame(const FrameDetections& f, const std::deque<FrameDetections>& buffer)
//   {
//       // Check number of detections
//       if (f.pixels.size() < min_detections_) {
//           RCLCPP_INFO(get_logger(), "Frame rejected, too few detections: %lu", f.pixels.size());
//           return false;
//       }

//       // Check pattern size in image
//       float xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
//       for (const auto& p : f.pixels) {
//           xmin = std::min(xmin, p.x);
//           xmax = std::max(xmax, p.x);
//           ymin = std::min(ymin, p.y);
//           ymax = std::max(ymax, p.y);
//       }

//       if ((xmax - xmin) < 0.15 * image_width_ || 
//           (ymax - ymin) < 0.15 * image_height_) {
//           RCLCPP_INFO(get_logger(), "Frame rejected, pattern too small");
//           return false;
//       }

//       // Check displacement against last frame in buffer
//       if (!buffer.empty() && !isDifferentEnough(buffer.back(), f)) {
//           RCLCPP_INFO(get_logger(), "Frame rejected, static pattern");
//           return false;
//       }

//       return true;
//   }


//   bool isDifferentEnough(const FrameDetections& last, const FrameDetections& current)
//   {
//       float meanDist = 0.0f;
//       int n = 0;

//       // On compare uniquement les coins dont les IDs sont communs
//       for (size_t i = 0; i < current.ids.size(); ++i)
//       {
//           auto it = std::find(last.ids.begin(), last.ids.end(), current.ids[i]);
//           if (it != last.ids.end())
//           {
//               size_t j = std::distance(last.ids.begin(), it);
//               float dx = current.pixels[i].x - last.pixels[j].x;
//               float dy = current.pixels[i].y - last.pixels[j].y;
//               meanDist += std::sqrt(dx*dx + dy*dy);
//               n++;
//           }
//       }

//       // Si pas de coins communs, considérer comme différent
//       if (n == 0) 
//           return true;

//       // Moyenne de la distance des coins communs
//       meanDist /= n;

//       // Seuil minimal de déplacement en pixels
//       const float min_move_pixels = 20.0f; // ajustable selon la résolution
//       return meanDist >= min_move_pixels;
//   }


//   // ---------------- Stereo builder ----------------

//   void tryBuildStereo(const FrameDetections& f, std::deque<FrameDetections>& other_buffer)
//   {
//       if (other_buffer.empty()) return;

//       // Cherche la frame la plus proche en timestamp
//       const FrameDetections* nearest = nullptr;
//       double best_dt = std::numeric_limits<double>::max();      
//       for (const auto& bf : other_buffer)
//       {
//           double dt = std::fabs((bf.stamp - f.stamp).seconds());
//           if (dt < best_dt)
//           {
//               best_dt = dt;
//               nearest = &bf;
//           }
//       }
      
//       if (nearest && best_dt < max_dt_)
//       {
//           StereoFrame sf;
//           if (f.stamp < nearest->stamp)
//           {
//               sf.cam0 = f;
//               sf.cam1 = *nearest;
//           }
//           else
//           {
//               sf.cam0 = *nearest;
//               sf.cam1 = f;
//           }
//           stereo_frames_.push_back(sf);
//           RCLCPP_INFO(get_logger(), "Stereo frames: %lu, dt=%.4f", stereo_frames_.size(), best_dt);
//       }

//       // Nettoyer les frames trop vieilles dans l'autre buffer
//       while (!other_buffer.empty() && (f.stamp - other_buffer.front().stamp).seconds() > max_dt_) {
//           other_buffer.pop_front();
// }
//   }

//   StereoFrame buildStereo(const FrameDetections& f0,
//                           const FrameDetections& f1)
//   {
//     StereoFrame sf;
//     sf.cam0.stamp = f0.stamp;
//     sf.cam1.stamp = f1.stamp;

//     for (size_t i = 0; i < f0.ids.size(); ++i)
//     {
//       int id = f0.ids[i];

//       for (size_t j = 0; j < f1.ids.size(); ++j)
//       {
//         if (f1.ids[j] == id)
//         {
//           sf.cam0.ids.push_back(id);
//           sf.cam0.pixels.push_back(f0.pixels[i]);

//           sf.cam1.ids.push_back(id);
//           sf.cam1.pixels.push_back(f1.pixels[j]);
//         }
//       }
//     }

//     return sf;
//   }

//   // ---------------- Callbacks ----------------

// void cbCam0(const AprilTagArray::SharedPtr msg)
// {
//     FrameDetections f0 = getDetectionsFromMsg(msg);

//     if (goodFrame(f0, cam0_buffer_)) {
//         mono_cam0_frames_.push_back(f0);
//         cam0_buffer_.push_back(f0);
//         RCLCPP_INFO(get_logger(), "Cam0 mono frames: %lu", mono_cam0_frames_.size());
//         tryBuildStereo(f0, cam1_buffer_);
//     }

    
// }


//   void cbCam1(const AprilTagArray::SharedPtr msg)
//   {
//       FrameDetections f1 = getDetectionsFromMsg(msg);

//       if (goodFrame(f1, cam1_buffer_)) {
//           mono_cam1_frames_.push_back(f1);
//           cam1_buffer_.push_back(f1);
//           RCLCPP_INFO(get_logger(), "Cam1 mono frames: %lu", mono_cam1_frames_.size());

//           tryBuildStereo(f1, cam0_buffer_);    
//       }
//   }
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<AprilGridCalibrationNode>());
//   rclcpp::shutdown();
//   return 0;
// }
