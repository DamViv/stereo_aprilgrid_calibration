#ifndef ACALIBRATOR_HPP
#define ACALIBRATOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "aprilgrid_calibration_core/projections_double_sphere.hpp"
#include "aprilgrid_calibration_core/camera_model.hpp"

class ACalibrator
{
public:
    virtual ~ACalibrator() = default;

    virtual bool ready() const = 0;
    virtual bool calibrate() = 0;

    CameraModel model() const { return model_; }
    cv::Size imageSize() const { return image_size_; }
    
    double reprojectionError() const {return reproj_error_; }

protected:

    void displayIntrinsics(double* intrinsics, CameraModel model){
        if(model == CameraModel::DOUBLE_SPHERE){
            std::cout << "Intrinsics:" << std::endl;
            std::cout << "fx_0  = " << intrinsics[0] << std::endl;
            std::cout << "fy_0  = " << intrinsics[1] << std::endl;
            std::cout << "cx_0  = " << intrinsics[2] << std::endl;
            std::cout << "cy_0  = " << intrinsics[3] << std::endl;
            std::cout << "xi    = " << intrinsics[4] << std::endl;
            std::cout << "alpha = " << intrinsics[5] << std::endl;
        }
    };

    void savePointsToCSV(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points)
    {
        std::ofstream file("correspondances.csv");
        if (!file.is_open()) {
            std::cerr << "Impossible d'ouvrir le fichier correspondances.csv" << std::endl;
            return;
        }
        file << "img_idx,pt_idx,X,Y,Z,u,v\n";
        for (size_t i = 0; i < object_points.size(); ++i) {
            for (size_t j = 0; j < object_points[i].size(); ++j) {
                const auto& p3d = object_points[i][j];
                const auto& p2d = image_points[i][j];
                file << i << "," << j << ","
                    << std::setprecision(10) << p3d.x << "," << p3d.y << "," << p3d.z << ","
                    << p2d.x << "," << p2d.y << "\n";
            }
        }
        file.close();
        std::cout << "Points sauvegardés dans correspondances.csv" << std::endl;
    };
    
    void savePointsToCSV(
        const std::vector<std::vector<cv::Point3f>>& object_points,
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const std::vector<std::vector<cv::Point2f>>& image_points1)
    {
        std::ofstream file("correspondances.csv");
        if (!file.is_open()) {
            std::cerr << "Impossible d'ouvrir le fichier correspondances.csv" << std::endl;
            return;
        }
        file << "img_idx,pt_idx,X,Y,Z,u,v,u1,v1\n";

        for (size_t i = 0; i < object_points.size(); ++i) {
            for (size_t j = 0; j < object_points[i].size(); ++j) {
                const auto& p3d = object_points[i][j];
                const auto& p2d = image_points[i][j];
                const auto& p2d1 = image_points1[i][j];
                file << i << "," << j << ","
                    << std::setprecision(10) << p3d.x << "," << p3d.y << "," << p3d.z << ","
                    << p2d.x << "," << p2d.y << "," << p2d1.x << "," << p2d1.y << "\n";
            }
        }
        file.close();
        std::cout << "Points sauvegardés dans correspondances.csv" << std::endl;
    };


    ACalibrator(int image_width, int image_height, CameraModel model)
        : image_size_(image_width, image_height), model_(model) {}

    cv::Size image_size_;
    CameraModel model_;
    double reproj_error_ = -1.0;
};


#endif // ACALIBRATOR_HPP