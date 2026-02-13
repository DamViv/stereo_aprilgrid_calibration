#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <pcl_conversions/pcl_conversions.h>



#include "../include/denseStereo.hpp"
#include "../include/sphericRectification.hpp"

class DoubleSphereCorrector : public rclcpp::Node
{
public:
    DoubleSphereCorrector() : Node("DoubleSphereCorrector")
    {
        // Image subscriptions
        this->declare_parameter<std::string>("img_left_topic", "/cam_0/image_raw");
        this->declare_parameter<std::string>("img_right_topic", "/cam_1/image_raw");
        this->get_parameter("img_left_topic", img_left_topic_);
        this->get_parameter("img_right_topic", img_right_topic_);

        image_sub1_ = create_subscription<sensor_msgs::msg::Image>(
                img_left_topic_, 1, std::bind(&DoubleSphereCorrector::imageCallback1, this, std::placeholders::_1));
        image_sub2_ = create_subscription<sensor_msgs::msg::Image>(
                img_right_topic_, 1, std::bind(&DoubleSphereCorrector::imageCallback2, this, std::placeholders::_1));

        // Double sphere parameters
        this->declare_parameter<double>("vfov", 90.);
        this->declare_parameter<int>("ndisp", 128);
        this->declare_parameter<int>("wsize", 7);
        this->declare_parameter<double>("min_dist", 0.5);
        this->declare_parameter<double>("max_dist", 10.);
        this->get_parameter("vfov", vfov_);
        this->get_parameter("ndisp", ndisp_);
        this->get_parameter("wsize", wsize_);

        // Create double sphere object
        this->declare_parameter<std::string>("stereo_config_file", "stereo_config_file.yaml");
        this->get_parameter("stereo_config_file", stereo_config_file_);
        RCLCPP_INFO(this->get_logger(), "Stereo config file = %s",
            stereo_config_file_.c_str());
        _ds = std::make_shared<denseStereo>(stereo_config_file_);
        _ds->_vfov        = vfov_;
        _ds->_ndisp       = ndisp_;
        _ds->_wsize       = wsize_;
        _ds->InitRectifyMap();

        _sr = std::make_shared<sphericRectification>(stereo_config_file_);

        // publishers
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/stereo/pointcloud", 10);
        depth_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/stereo/depth_image", 10);
    }


    

private:
    std::string img_left_topic_, img_right_topic_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub1_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub2_;
    cv::Mat last_left_img, last_right_img;
    
    std::vector<std::pair<rclcpp::Time, cv::Mat>> left_imgs_, right_imgs_;

    std::shared_ptr<denseStereo> _ds;
    std::shared_ptr<sphericRectification> _sr;
    std::string stereo_config_file_;    
    double vfov_;
    int ndisp_, wsize_;

    double max_time_delta_ = 0.1;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;


    void imageCallback1(const sensor_msgs::msg::Image &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "mono8");
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        last_left_img = cv_ptr->image;   
        
        std::pair<rclcpp::Time, cv::Mat> current(msg.header.stamp, last_left_img.clone());
        if(!tryBuildStereo(current, right_imgs_, true))
            left_imgs_.emplace_back(current);
    }


    void imageCallback2(const sensor_msgs::msg::Image &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "mono8");
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        last_right_img = cv_ptr->image;
        
        std::pair<rclcpp::Time, cv::Mat> current(msg.header.stamp, last_right_img.clone());
        if(!tryBuildStereo(current, left_imgs_, false))
            right_imgs_.emplace_back(current);
    }


    bool tryBuildStereo(std::pair<rclcpp::Time, cv::Mat>& f,
                        std::vector<std::pair<rclcpp::Time, cv::Mat>>& other_buffer, bool left){
        if (other_buffer.empty())
            return false;
                   
        double best_dt = std::numeric_limits<double>::max();          
        size_t best_index = 0;

        for (size_t i = 0; i < other_buffer.size(); ++i) {
            double dt = std::fabs((other_buffer[i].first - f.first).seconds());
            if (dt < best_dt) {
                best_dt = dt;
                best_index = i;
            }
        }

        if (best_dt < max_time_delta_) {
            auto matched = other_buffer[best_index];
            if (left)
                processStereo3D(f.second, matched.second);
            else
                processStereo3D(matched.second, f.second);        
            other_buffer.erase(other_buffer.begin() + best_index);
            return true;
        }

        return false;
    }


    void processStereo3D(cv::Mat left, cv::Mat right){
        
        cv::imshow("left", left);
        cv::imshow("right", right);
        cv::waitKey(1);        

        _sr->processImages(left, right);
        return;
        
        // Downsample image
        cv::Mat small_left_img, small_right_img;
        cv::resize(left, small_left_img, cv::Size(), 1, 1);
        cv::resize(right, small_right_img, cv::Size(), 1, 1);

        cv::imshow("left", left);

        cv::Mat rect_imgl, rect_imgr;
        cv::remap(small_left_img, rect_imgl, _ds->smap[0][0], _ds->smap[0][1], 1, 0);
        cv::remap(small_right_img, rect_imgr, _ds->smap[1][0], _ds->smap[1][1], 1, 0);

        cv::imshow("rect_imgl", rect_imgl);
        cv::waitKey(1);

        cv::imshow("rect_imgr", rect_imgr);
        cv::waitKey(1);

        // // Disparity computation
        cv::Mat disp_img, depth_map;
        _ds->DisparityImage(rect_imgl, rect_imgr, disp_img, depth_map);

        // // Depth image filtering
        cv::Mat depth_filtered;
        cv::medianBlur(depth_map, depth_filtered, 5);

        // // Pointcloud computation
        // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_stereo = _ds->pcFromDepthMap(depth_filtered);

        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_stereo = _ds->Triangulate(rect_imgl, rect_imgr);

        // Generate and send message
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*pcl_cloud_stereo, cloud_msg);
        cloud_msg.header.frame_id = "camera_frame";
        cloud_msg.header.stamp = this->get_clock()->now();
        pointcloud_pub_->publish(cloud_msg);

        cv_bridge::CvImage cv_img;
        cv_img.header.stamp = this->get_clock()->now();
        cv_img.header.frame_id = "camera_frame";
        cv_img.encoding = "32FC1";
        cv_img.image = depth_filtered;
        depth_img_pub_->publish(*cv_img.toImageMsg());

    }

};



// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<DoubleSphereCorrector>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("calibration_manager"), 
                    "Fatal error: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}