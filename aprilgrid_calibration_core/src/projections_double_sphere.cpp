#include "../include/aprilgrid_calibration_core/projections_double_sphere.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Assumes you have a DSCamera class with world2cam method
// bool DSCamera::world2cam(const cv::Vec3f& point3D, cv::Vec2f& uv);

void showDSCorrected(const cv::Mat& img, const cv::Mat& K, const cv::Mat &D, const double &fov,
                     const std::string& mode,
                     cv::Size out_size,
                     float f_persp)
{
    cv::Mat mapX, mapY;
    mapX = cv::Mat(out_size, CV_32F);
    mapY = cv::Mat(out_size, CV_32F);

    int h = out_size.height;
    int w = out_size.width;

    for(int r=0; r<h; r++){
        for(int c=0; c<w; c++){
            cv::Vec3d ray;

            if(mode == "perspective"){
                float z = f_persp * std::min(w,h);
                float x = c - w/2.0f;
                float y = r - h/2.0f;
                ray = cv::Vec3d(x, y, z);
                ray = ray / cv::norm(ray);
            }
            else if(mode == "equirect"){
                float theta = -M_PI/2.0 + (r+0.5f) * M_PI / h;
                float phi   = -M_PI   + (c+0.5f) * 2*M_PI / w;
                float x = std::sin(phi) * std::cos(theta);
                float y = std::sin(theta);
                float z = std::cos(phi) * std::cos(theta);
                ray = cv::Vec3d(x, y, z);
            }
            else{
                std::cerr << "Unknown mode " << mode << std::endl;
                return;
            }

            cv::Vec2d uv;
            double intrinsics[6] = {K.at<double>(0,0), K.at<double>(1,1), K.at<double>(0,2), K.at<double>(1,2), D.at<double>(0,0), D.at<double>(0,1)};
            projectDoubleSphere(ray[0], ray[1], ray[2], intrinsics, uv[0], uv[1]);
                        
            mapX.at<float>(r,c) = static_cast<float>(uv[0]);
            mapY.at<float>(r,c) = static_cast<float>(uv[1]);

            
        }
    }

    cv::Mat corrected;
    cv::remap(img, corrected, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    cv::imshow("DS Corrected", corrected);
    cv::waitKey(1);
}