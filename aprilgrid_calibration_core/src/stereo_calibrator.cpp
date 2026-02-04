#include "aprilgrid_calibration_core/stereo_calibrator.hpp"
// #include "aprilgrid_calibration_core/projections_double_sphere.hpp" 
// #include <ceres/ceres.h>
// #include <ceres/rotation.h>
// #include <array>
// #include <iostream>

// #include <fstream>
// #include <iomanip>



void StereoCalibrator::setIntrinsics(const cv::Mat& K0, const cv::Mat& D0, 
                                      const cv::Mat& K1, const cv::Mat& D1){ 
    K0.copyTo(K0_);
    D0.copyTo(D0_); 
    K1.copyTo(K1_); 
    D1.copyTo(D1_); 
};


void StereoCalibrator::initIntrinsics(const cv::Mat& K,  const cv::Mat& D, double intrinsics[6]){
    
    if (model_ == CameraModel::DOUBLE_SPHERE) {        

        if(K.empty()) {
            intrinsics[0] = image_size_.width;
            intrinsics[1] = image_size_.height;
            intrinsics[2] = image_size_.width/2.0;
            intrinsics[3] = image_size_.height/2.0;
            intrinsics[4] = 0.5;
            intrinsics[5] = 0.5;    
        }
        else {
            intrinsics[0] = K.at<double>(0,0);
            intrinsics[1] = K.at<double>(1,1);
            intrinsics[2] = K.at<double>(0,2);
            intrinsics[3] = K.at<double>(1,2);
            intrinsics[4] = D.at<double>(0);
            intrinsics[5] = D.at<double>(1);    
        }
    }    
}



bool StereoCalibrator::calibrate() {
    if (!ready()) return false;

    // init intrinsics
    double intrinsics0[6];
    initIntrinsics(K0_, D0_, intrinsics0);
    double intrinsics1[6];
    initIntrinsics(K1_, D1_, intrinsics1);
    
    if (model_ == CameraModel::DOUBLE_SPHERE) {    
        // Initialisation des poses caméra (RT) par vue
        std::vector<std::array<double,6>> RT_mire_views(object_points_.size(), {0,0,0,0,0,0});        
        std::array<double,6> RT_cam1_cam0 = {0,0,0,0.2,0,0}; // initialement alignées
        
        return calibrateStereoDoubleSphere(intrinsics0, intrinsics1, RT_mire_views, RT_cam1_cam0, false);
    }

    // TODO Fisheye / Pinhole
    return false;
}


bool StereoCalibrator::calibrateStereoDoubleSphere(
    double* intrinsics0,
    double* intrinsics1,
    std::vector<std::array<double,6>>& RT_mire_views,    
    std::array<double,6>& RT_cam1_cam0,
    bool optimize_intrinsics)
{

    ceres::Problem problem;

    size_t n_views = object_points_.size();
    if (RT_mire_views.size() != n_views) {
        std::cerr << "[StereoCalib] RT_mire_views size mismatch\n";
        return false;
    }

    problem.AddParameterBlock(RT_cam1_cam0.data(), 6);
    problem.AddParameterBlock(intrinsics0, 6);
    problem.AddParameterBlock(intrinsics1, 6);

    for (size_t i=0; i<n_views; ++i) {
        const auto& pts3D = object_points_[i];
        const auto& pts0 = image_0_points_[i];
        const auto& pts1 = image_1_points_[i];

        if (pts3D.size() != pts0.size() || pts0.size() != pts1.size()) {
            std::cerr << "[StereoCalib] size mismatch view " << i << "\n";
            return false;
        }

        for (size_t j=0; j<pts3D.size(); ++j) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ReprojectionErrorStereoDS, 4, 6, 6, 6, 6>(
                    new ReprojectionErrorStereoDS(pts3D[j], pts0[j], pts1[j]));


            // utiliser Huber pour robustesse
            problem.AddResidualBlock(cost_function,
                                     new ceres::HuberLoss(1.0),
                                     intrinsics0,
                                     intrinsics1,
                                     RT_mire_views[i].data(),
                                     RT_cam1_cam0.data());
        }
    }


    // DOES NOT WORK IF INTRINSEC ARE CALIBRATED SIMULATANEOUSLY
    if (!optimize_intrinsics) {
        problem.SetParameterBlockConstant(intrinsics0);
        problem.SetParameterBlockConstant(intrinsics1);
    }

    // Setup Ceres solver
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 500;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-6;
    options.num_threads                        = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    double rms = std::sqrt(2.0 * summary.final_cost / summary.num_residuals);    
    std::cout << "Reprojection RMSE = " << rms << " pixels" << std::endl;

    std::cout << "Calibrated intrinsics cam 0:" << std::endl;
    displayIntrinsics(intrinsics0, model_);
   
    std::cout << "Calibrated intrinsics cam 0:" << std::endl;
    displayIntrinsics(intrinsics1, model_);

    std::cout << "Extrinsics cam1->cam0: rotation (angle-axis) = [" 
              << RT_cam1_cam0[0] << "," << RT_cam1_cam0[1] << "," << RT_cam1_cam0[2] << "], translation = [" 
              << RT_cam1_cam0[3] << "," << RT_cam1_cam0[4] << "," << RT_cam1_cam0[5] << "]" << std::endl;


    // Stockage des résultats de calibration
    K0_ = (cv::Mat_<double>(3,3) << intrinsics0[0], 0, intrinsics0[2],
                                    0, intrinsics0[1], intrinsics0[3],
                                    0, 0, 1);
    D0_ = (cv::Mat_<double>(6,1) << intrinsics0[4], intrinsics0[5], 0, 0, 0, 0);

    K1_ = (cv::Mat_<double>(3,3) << intrinsics1[0], 0, intrinsics1[2],
                                    0, intrinsics1[1], intrinsics1[3],
                                    0, 0, 1);
    D1_ = (cv::Mat_<double>(6,1) << intrinsics1[4], intrinsics1[5], 0, 0, 0, 0);    

    RT_stereo_ = RT_cam1_cam0;

    reproj_error_ = rms;


    return summary.termination_type == ceres::CONVERGENCE;
}
