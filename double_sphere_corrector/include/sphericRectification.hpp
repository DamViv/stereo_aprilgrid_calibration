#include <chrono>
#include <cmath>
#include <iostream>
#include <stack>

// For opencv
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// For pcl generation
#include "pcl/common/transforms.h"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>



class sphericRectification {
  public:
    sphericRectification() {}
    sphericRectification(std::string configfilepath);

    void loadStereoConfig(const std::string& config_file);

    void processImages(const cv::Mat& Il, const cv::Mat& Ir);

 private:  

    void computeSphericalMaps(int width, int height, 
        cv::Mat& map_x_r, cv::Mat& map_y_r,
        cv::Mat& map_x_l, cv::Mat& map_y_l,        
        double FOVx_deg, double FOVy_deg);


    cv::Vec3d unprojectDS(const cv::Vec2d& uv, const cv::Mat& K, double xi, double alpha);
    cv::Vec2d projectDS(const cv::Vec3d& ray, const cv::Mat& K, double xi, double alpha);
    

    cv::Mat Kl_, Kr_;
    double xil_, xir_, alphal_, alphar_;
    cv::Mat Rstereo_, Tstereo_;
    cv::Size img_size_;
};