#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "opencv2/opencv.hpp"
#include <cmath>

// April tags detector and various families that can be selected by command line option
#include "../include/apriltags/TagDetector.h"
#include "../include/apriltags/Tag16h5.h"
#include "../include/apriltags/Tag25h7.h"
#include "../include/apriltags/Tag25h9.h"
#include "../include/apriltags/Tag36h9.h"
#include "../include/apriltags/Tag36h11.h"

#include "aprilgrid_detector_interfaces/msg/april_tag.hpp"
#include "aprilgrid_detector_interfaces/msg/april_tag_array.hpp"
#include "aprilgrid_detector_interfaces/srv/start_collection.hpp"
#include "aprilgrid_detector_interfaces/srv/stop_collection.hpp"




const char* window_name = "apriltags_detector_";

#ifndef PI
    const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;

inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
  }






class AprilGridDetectorNode : public rclcpp::Node {

public:
    AprilGridDetectorNode() : Node("aprilgrid_detector_ros2"),
        tagCodes_(AprilTags::tagCodes36h11) 
    {

        // ===== Déclaration des paramètres =====        
        this->declare_parameter<std::string>("image_topic", "/cam_0/image_raw,/cam_1/image_raw");
        this->declare_parameter<std::string>("output_topic", "/cam_0/apriltag, /cam_1/apriltag");
        this->declare_parameter<std::string>("tag_family", "36h11");
        this->declare_parameter<double>("tag_size", 0.5);

        this->declare_parameter<int>("width", 640);
        this->declare_parameter<int>("height", 480);
        this->declare_parameter<double>("fx", 600.0);
        this->declare_parameter<double>("fy", 600.0);
        this->declare_parameter<double>("px", 320.0);
        this->declare_parameter<double>("py", 240.0);

        this->declare_parameter<bool>("display", true);

        // ===== Lecture des paramètres =====
        std::string tag_family;        
        std::string image_topics_str, output_topics_str;
        this->get_parameter("image_topic", image_topics_str);
        this->get_parameter("output_topic", output_topics_str);
        
        std::vector<std::string> input_topics;
        std::vector<std::string> output_topics;    
        std::stringstream ss(image_topics_str);
        std::string topic;
        while (std::getline(ss, topic, ',')) input_topics.push_back(topic);
        std::stringstream ss2(output_topics_str);
        std::string topic2;
        while (std::getline(ss2, topic2, ',')) output_topics.push_back(topic2);

        setup_cameras(input_topics, output_topics);
        
        this->get_parameter("tag_family", tag_family);
        this->get_parameter("tag_size", tagSize_);

        this->get_parameter("width", width_);
        this->get_parameter("height", height_);
        this->get_parameter("fx", fx_);
        this->get_parameter("fy", fy_);
        this->get_parameter("px", px_);
        this->get_parameter("py", py_);

        this->get_parameter("display", display_);

        // ===== Tag family =====
        setTagCodes(tag_family);
        tagDetector_ = std::make_shared<AprilTags::TagDetector>(tagCodes_);

        RCLCPP_INFO(this->get_logger(), "AprilGrid detector started");        
        RCLCPP_INFO(this->get_logger(), " Tag family: %s", tag_family.c_str());
        RCLCPP_INFO(this->get_logger(), " Tag size: %.3f m", tagSize_);

        // ===== Services activation ======
        start_service_ = this->create_service<aprilgrid_detector_interfaces::srv::StartCollection>(
            "/calibration/start_collection",
            [this](const std::shared_ptr<aprilgrid_detector_interfaces::srv::StartCollection::Request> request,
                std::shared_ptr<aprilgrid_detector_interfaces::srv::StartCollection::Response> response)
            {
                switch (request->mode) {
                    case 1: // MODE_MONO_CAM0
                        collecting_cam0_ = true;
                        break;
                    case 2: // MODE_MONO_CAM1
                        collecting_cam1_ = true;
                        break;
                    case 3: // MODE_STEREO
                        collecting_stereo_ = true;                        
                        break;
                    case 0: // MODE_IDLE
                    default:
                        // ne rien faire
                        break;
                }

                RCLCPP_INFO(this->get_logger(), "StartCollection called: collecting_cam0=%s, collecting_cam1=%s, collecting_stereo=%s",
                            collecting_cam0_ ? "true" : "false",
                            collecting_cam1_ ? "true" : "false",
                            collecting_stereo_ ? "true" : "false");
                response->success = true;
                response->message = "Collection started";
            }
        );

        stop_service_ = this->create_service<aprilgrid_detector_interfaces::srv::StopCollection>(
            "/calibration/stop_collection",
            [this](const std::shared_ptr<aprilgrid_detector_interfaces::srv::StopCollection::Request> request,
                std::shared_ptr<aprilgrid_detector_interfaces::srv::StopCollection::Response> response)
            {
                switch (request->mode) {
                    case 1: // MODE_MONO_CAM0
                        collecting_cam0_ = false;
                        break;
                    case 2: // MODE_MONO_CAM1
                        collecting_cam1_ = false;
                        break;
                    case 3: // MODE_STEREO
                        collecting_stereo_ = false;                        
                        break;
                    case 0: // MODE_IDLE
                    default:
                        // ne rien faire
                        break;
                }

                RCLCPP_INFO(this->get_logger(), "StopCollection called: collecting_cam0=%s, collecting_cam1=%s, collecting_stereo=%s",
                            collecting_cam0_ ? "true" : "false",
                            collecting_cam1_ ? "true" : "false",
                            collecting_stereo_ ? "true" : "false");
                response->success = true;
                response->message = "Collection stoped";
            }
        );



    }

private:
    bool display_ = false;
    int width_, height_;
    double fx_, fy_, px_, py_;
    double tagSize_;    

    // ===== Services definition ======
    rclcpp::Service<aprilgrid_detector_interfaces::srv::StartCollection>::SharedPtr start_service_;
    rclcpp::Service<aprilgrid_detector_interfaces::srv::StopCollection>::SharedPtr stop_service_;    

    // Flags pour savoir quelles caméras collecter
    bool collecting_cam0_ = false;
    bool collecting_cam1_ = false;
    bool collecting_stereo_ = false;

    struct CameraHandler {
        std::string image_topic;
        std::string output_topic;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub;
        rclcpp::Publisher<aprilgrid_detector_interfaces::msg::AprilTagArray>::SharedPtr pub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub;

    };

    std::vector<CameraHandler> aprilgrid_;    
    std::shared_ptr<AprilTags::TagDetector> tagDetector_;
    AprilTags::TagCodes tagCodes_;


    void setup_cameras(std::vector<std::string> input_topics, std::vector<std::string> output_topics){
        auto sensor_qos = rclcpp::SensorDataQoS();

        for (size_t i = 0; i < input_topics.size(); i++) {
            CameraHandler cam;
            cam.image_topic = input_topics[i];
            cam.output_topic = output_topics[i];

            cam.sub = this->create_subscription<sensor_msgs::msg::Image>(
                cam.image_topic,
                sensor_qos,
                [this, i](sensor_msgs::msg::Image::ConstSharedPtr msg)
                {
                    this->imageCallback(msg, i);
                }
            );

            cam.pub = this->create_publisher<aprilgrid_detector_interfaces::msg::AprilTagArray>(cam.output_topic, sensor_qos);
            cam.annotated_pub = this->create_publisher<sensor_msgs::msg::Image>(cam.output_topic + "_img", sensor_qos);

            aprilgrid_.push_back(cam);

        }
    }


    void setTagCodes(string s) {
        if (s=="16h5") {
        tagCodes_ = AprilTags::tagCodes16h5;
        } else if (s=="25h7") {
        tagCodes_ = AprilTags::tagCodes25h7;
        } else if (s=="25h9") {
        tagCodes_ = AprilTags::tagCodes25h9;
        } else if (s=="36h9") {
        tagCodes_ = AprilTags::tagCodes36h9;
        } else if (s=="36h11") {
        tagCodes_ = AprilTags::tagCodes36h11;
        } else {
        cout << "Invalid tag family specified" << endl;
        exit(1);
        }
    }


  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg, size_t cam_index)
  {
    
    // RCLCPP_INFO(this->get_logger(),
    //     "Callback cam=%zu | cam0=%d cam1=%d stereo=%d",
    //     cam_index, collecting_cam0_, collecting_cam1_, collecting_stereo_);

    if(!((collecting_cam0_ && cam_index == 0) || (collecting_cam1_ && cam_index == 1) || collecting_stereo_))
        return;
    
    cv::Mat image;
    cv::Mat image_gray;

    try {
      image = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }


    if (image.channels() == 3) {
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    } else {
        image_gray = image;
    }    

    std::vector<AprilTags::TagDetection> detections = tagDetector_->extractTags(image_gray);

    if(display_) {
        cout << "For camera " << cam_index << ": " << detections.size() << " tags detected:" << endl;
        for (size_t i = 0; i < detections.size(); i++) {
            print_detection(detections[i]);
        }
    }

    for (size_t i = 0; i < detections.size(); i++) {
        detections[i].draw(image);
    }

    if (display_) {
      cv::imshow(window_name + std::to_string(cam_index), image);
      cv::waitKey(1);
    }

    publishDetections(detections, msg->header, aprilgrid_[cam_index].pub);
    
    // --- Publish the annotated image ---
    auto annotated_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
    aprilgrid_[cam_index].annotated_pub->publish(*annotated_msg);
}


void publishDetections(
    const std::vector<AprilTags::TagDetection> & detections,
    const std_msgs::msg::Header & header,
    rclcpp::Publisher<aprilgrid_detector_interfaces::msg::AprilTagArray>::SharedPtr pub)
{
    aprilgrid_detector_interfaces::msg::AprilTagArray array_msg;
    array_msg.header = header;

    for (const auto & det : detections) {
        aprilgrid_detector_interfaces::msg::AprilTag tag_msg;
        tag_msg.id = det.id;
        tag_msg.hamming = det.hammingDistance;

        // centre
        tag_msg.center = {det.cxy.first, det.cxy.second};

        // coins
        //tag_msg.corners.resize(8);
        for (int i=0; i<4; i++){
            tag_msg.corners[i*2]   = det.p[i].first;
            tag_msg.corners[i*2+1] = det.p[i].second;
        }

        array_msg.tags.push_back(tag_msg);
    }

    pub->publish(array_msg);
}



void print_detection(AprilTags::TagDetection& detection) const {

    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    detection.getRelativeTranslationRotation(tagSize_, fx_, fy_, px_, py_,
                                             translation, rotation);

    Eigen::Matrix3d F;
    F <<
      1, 0,  0,
      0,  -1,  0,
      0,  0,  1;
    Eigen::Matrix3d fixed_rot = F*rotation;
    double yaw, pitch, roll;
    wRo_to_euler(fixed_rot, yaw, pitch, roll);

    cout << "  distance=" << translation.norm()
         << "m, x=" << translation(0)
         << ", y=" << translation(1)
         << ", z=" << translation(2)
         << ", yaw=" << yaw
         << ", pitch=" << pitch
         << ", roll=" << roll
         << endl;
  }

};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AprilGridDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
