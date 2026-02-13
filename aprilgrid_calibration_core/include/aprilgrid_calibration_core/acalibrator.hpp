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
#include <optional>

#include "aprilgrid_calibration_core/camera_model.hpp"


struct Extrinsics {
    cv::Mat R;
    cv::Mat t;
};



class ACalibrator
{
public:
    virtual ~ACalibrator() = default;

    virtual bool ready() const = 0;
    virtual bool calibrate() = 0;

    CameraModel model() const { return model_; }
    cv::Size imageSize() const { return image_size_; }

    void setIntrinsics(size_t ind, const cv::Mat& K, const cv::Mat& D){
        K_.at(ind) = K.clone();
        D_.at(ind) = D.clone();
    }

    const cv::Mat& cameraIntrinsics(size_t ind) const {
        return K_.at(ind);
    }

    const cv::Mat& distCoeffs(size_t ind) const {
        return D_.at(ind);
    }

    virtual std::optional<Extrinsics> getExtrinsics() const = 0;

    double reprojectionError() const {return reproj_error_; }

protected:

    std::vector<cv::Mat> K_{2};
    std::vector<cv::Mat> D_{2};   



    double computeRMSE(const ceres::Solver::Summary& summary)
    {
        if (summary.num_residuals == 0) return std::numeric_limits<double>::infinity();
        return std::sqrt(2.0 * summary.final_cost / summary.num_residuals);
    }


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

    void displayIntrinsics(double* intrinsics, CameraModel model, double* cov){
        if(model == CameraModel::DOUBLE_SPHERE){
            std::cout << "Intrinsics:" << std::endl;
            std::cout << "fx_0  = " << intrinsics[0] << " +/- " << std::sqrt(cov[0]) <<  std::endl;
            std::cout << "fy_0  = " << intrinsics[1] << " +/- " << std::sqrt(cov[7]) <<  std::endl;
            std::cout << "cx_0  = " << intrinsics[2] << " +/- " << std::sqrt(cov[14]) <<  std::endl;
            std::cout << "cy_0  = " << intrinsics[3] << " +/- " << std::sqrt(cov[21]) <<  std::endl;
            std::cout << "xi    = " << intrinsics[4] << " +/- " << std::sqrt(cov[28]) <<  std::endl;
            std::cout << "alpha = " << intrinsics[5] << " +/- " << std::sqrt(cov[35]) <<  std::endl;
        }
    };


    void displayExtrinsics(std::array<double,6> extrinsics){
        
        std::cout << "Extrinsics:" << std::endl;
        std::cout << "rx  = " << extrinsics[0] << std::endl;
        std::cout << "ry  = " << extrinsics[1] << std::endl;
        std::cout << "rz  = " << extrinsics[2] << std::endl;
        std::cout << "tx  = " << extrinsics[3] << std::endl;
        std::cout << "ty  = " << extrinsics[4] << std::endl;
        std::cout << "tz  = " << extrinsics[5] << std::endl;

    };

    void displayExtrinsics(std::array<double,6> extrinsics, double* cov){        
        std::cout << "Extrinsics:" << std::endl;
        std::cout << "rx  = " << extrinsics[0] << " +/- " << std::sqrt(cov[0]) <<  std::endl;
        std::cout << "ry  = " << extrinsics[1] << " +/- " << std::sqrt(cov[7]) <<  std::endl;
        std::cout << "rz  = " << extrinsics[2] << " +/- " << std::sqrt(cov[14]) <<  std::endl;
        std::cout << "tx  = " << extrinsics[3] << " +/- " << std::sqrt(cov[21]) <<  std::endl;
        std::cout << "ty  = " << extrinsics[4] << " +/- " << std::sqrt(cov[28]) <<  std::endl;
        std::cout << "tz  = " << extrinsics[5] << " +/- " << std::sqrt(cov[35]) <<  std::endl;        
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




struct PriorResidual {
    PriorResidual(double mean, double sigma, int index) : mean(mean), sigma(sigma), index(index) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        // x = bloc entier (6 paramètres), on n'utilise que x[index]
        residual[0] = (x[index] - T(mean)) / T(sigma);
        return true;
    }

    double mean; // valeur cible
    double sigma; // incertitude
    int index;   // indice dans le bloc (0..5)
};



#endif // ACALIBRATOR_HPP