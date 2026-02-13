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



class denseStereo {
  public:
    denseStereo() {}
    denseStereo(std::string configfilepath);

    void loadStereoConfig(const std::string& config_file);

    void InitUndistortRectifyMap(cv::Mat K,
                                 cv::Mat D,
                                 double xi,
                                 cv::Mat R,
                                 cv::Mat P,
                                 cv::Size size,
                                 cv::Mat &map1,
                                 cv::Mat &map2);
    void InitUndistortRectifyMapDS(cv::Mat K,
                                 float alpha,
                                 float xi,
                                 cv::Mat R,
                                 cv::Mat P,
                                 cv::Size size,
                                 cv::Mat &map1,
                                 cv::Mat &map2);
    void InitRectifyMap();
    void DisparityImage(const cv::Mat &recl, const cv::Mat &recr, cv::Mat &disp, cv::Mat &depth_map);
    pcl::PointCloud<pcl::PointXYZ>::Ptr Triangulate(const cv::Mat &recl, const cv::Mat &recr);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcFromDepthMap(const cv::Mat &depth_map); 

    std::string _configfilepath, _cam_model;
    cv::Mat Translation, Kl, Kr, Dl, Dr, Rl, Rr, smap[2][2], Knew;
    double xil, xir, alphal, alphar;

    int _vfov = 60; 
    int _width, _height;

    int _ndisp = 32, _wsize = 7;

    int _cap_cols, _cap_rows;
    bool is_sgbm = true;
};