#include "../include/aprilgrid_calibration_core/mono_calibrator.hpp"





void MonoCalibrator::addView(const std::vector<cv::Point2f>& pts2D,
                             const std::vector<cv::Point3f>& pts3D)
{
    if (pts2D.size() != pts3D.size())
        return;

    if (pts2D.size() < 10)
        return;

    image_points_.push_back(pts2D);
    object_points_.push_back(pts3D);
}

bool MonoCalibrator::ready() const
{
    return image_points_.size() >= 10;
}


bool MonoCalibrator::calibrate()
{

    // Intrinsics initialisation
    double intrinsics[6] = {(double)image_size_.width, (double)image_size_.height, (double)image_size_.width/2., (double)image_size_.height/2., 0.5, 0.5};

    // Vecteur pour stocker les poses initiales (AngleAxis + translation)
    std::vector<std::array<double,6>> rts(object_points_.size());

    // --- Initialisation des poses avec solvePnP ---
    cv::Mat K = (cv::Mat_<double>(3,3) << (double)image_size_.width, 0, (double)image_size_.width/2., 0, (double)image_size_.height, (double)image_size_.height/2., 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_64F); // approximatif

    for (size_t i = 0; i < object_points_.size(); ++i) {
        cv::Mat rvec, tvec;

        // SolvePnP pour initialiser la pose
        bool ok = cv::solvePnP(object_points_[i], image_points_[i], K, distCoeffs, rvec, tvec);
        if (!ok) {
            std::cerr << "solvePnP failed for image " << i << std::endl;
            return false;
        }

        rts[i][0] = rvec.at<double>(0);
        rts[i][1] = rvec.at<double>(1);
        rts[i][2] = rvec.at<double>(2);
        rts[i][3] = tvec.at<double>(0);
        rts[i][4] = tvec.at<double>(1);
        rts[i][5] = tvec.at<double>(2);
    }


    if (model_ == CameraModel::PINHOLE)
    {
        // À implémenter si besoin
        std::cerr << "Pinhole model calibration not implemented yet." << std::endl;
        return false;
    }
    else if (model_ == CameraModel::FISHEYE)
    {        
        return calibrate_fisheye(intrinsics, rts);
    }
    else if (model_ == CameraModel::DOUBLE_SPHERE)
    {
        return calibrate_double_sphere(intrinsics, rts);
    }

    return false;
}   


bool MonoCalibrator::calibrate_pinhole(double* intrinsics, std::vector<std::array<double,6>>& rts){
    
    std::vector<cv::Mat> rvecs, tvecs;
    K_ = (cv::Mat_<double>(3,3) << intrinsics[0], 0, intrinsics[2],
                                0, intrinsics[1], intrinsics[3],
                                0, 0, 1);
    D_ = cv::Mat::zeros(4, 1, CV_64F);

    double rms = cv::calibrateCamera(
        object_points_,
        image_points_,
        image_size_,
        K_,
        D_,
        rvecs,
        tvecs,
        cv::CALIB_RATIONAL_MODEL
    );

    std::cout << "Calibrated intrinsics:" << std::endl;
    std::cout << "fx = " << K_.at<double>(0,0) << std::endl;
    std::cout << "fy = " << K_.at<double>(1,1) << std::endl;
    std::cout << "cx = " << K_.at<double>(0,2) << std::endl;
    std::cout << "cy = " << K_.at<double>(1,2) << std::endl;
    std::cout << "disto = " << D_.at<double>(0) << " " << D_.at<double>(1) << " " << D_.at<double>(2) << " " << D_.at<double>(3) << std::endl;    
    std::cout << "Reprojection RMSE = " << rms << " pixels" << std::endl;
    
    return true;
}


bool MonoCalibrator::calibrate_fisheye(double* intrinsics, std::vector<std::array<double,6>>& rts){
    
    std::vector<cv::Mat> rvecs, tvecs;
    K_ = (cv::Mat_<double>(3,3) << intrinsics[0], 0, intrinsics[2],
                                0, intrinsics[1], intrinsics[3],
                                0, 0, 1);
    D_ = cv::Mat::zeros(4, 1, CV_64F);
    
    int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                cv::fisheye::CALIB_CHECK_COND;

    double rms = cv::fisheye::calibrate(
        object_points_,
        image_points_,
        image_size_,
        K_,
        D_,
        rvecs,
        tvecs,
        flags
    );

    std::cout << "Calibrated intrinsics:" << std::endl;
    std::cout << "fx = " << K_.at<double>(0,0) << std::endl;
    std::cout << "fy = " << K_.at<double>(1,1) << std::endl;
    std::cout << "cx = " << K_.at<double>(0,2) << std::endl;
    std::cout << "cy = " << K_.at<double>(1,2) << std::endl;
    std::cout << "disto = " << D_.at<double>(0) << " " << D_.at<double>(1) << " " << D_.at<double>(2) << " " << D_.at<double>(3) << std::endl;    
    std::cout << "Reprojection RMSE = " << rms << " pixels" << std::endl;
    
    return true;
}

bool MonoCalibrator::calibrate_double_sphere(double* intrinsics, std::vector<std::array<double,6>>& rts)
{    
    ceres::Problem problem;   

    for (size_t i = 0; i < object_points_.size(); ++i) {
        double* rt = rts[i].data();
        for (size_t j = 0; j < object_points_[i].size(); ++j) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ReprojectionErrorDS, 2, 6, 6>(
                    new ReprojectionErrorDS(object_points_[i][j], image_points_[i][j]));

            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), intrinsics, rt);
        }
    }

    // --- Solveur Ceres ---
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY; // SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 500;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    double rms = std::sqrt(2.0 * summary.final_cost / summary.num_residuals);
    std::cout << "Reprojection RMSE = " << rms << " pixels" << std::endl;

    std::cout << "Calibrated intrinsics:" << std::endl;
    displayIntrinsics(intrinsics, model_);

    // // --- Affichage des poses finales ---
    // for (size_t i = 0; i < rts.size(); ++i) {
    //     double* rt = rts[i].data();
    //     std::cout << "Pose image " << i 
    //               << " : rotation (angle-axis) = [" 
    //               << rt[0] << "," << rt[1] << "," << rt[2] << "], translation = [" 
    //               << rt[3] << "," << rt[4] << "," << rt[5] << "]" << std::endl;
    // }

    // Stockage des résultats de calibration
    K_ = (cv::Mat_<double>(3,3) << intrinsics[0], 0, intrinsics[2],
                                    0, intrinsics[1], intrinsics[3],
                                    0, 0, 1);
    D_ = (cv::Mat_<double>(6,1) << intrinsics[4], intrinsics[5], 0, 0, 0, 0);
    reproj_error_ = rms;

    return summary.termination_type == ceres::CONVERGENCE;
}


// bool MonoCalibrator::calibrate()
// {
//     if (!ready())
//         return false;

//     printPointCorrespondences(object_points_, image_points_);

//     if (model_ == CameraModel::PINHOLE)
//     {
//         K_ = cv::Mat::eye(3, 3, CV_64F);
//         D_ = cv::Mat::zeros(8, 1, CV_64F);

//         std::vector<cv::Mat> rvecs, tvecs;

//         reproj_error_ = cv::calibrateCamera(
//             object_points_,
//             image_points_,
//             image_size_,
//             K_,
//             D_,
//             rvecs,
//             tvecs,
//             cv::CALIB_RATIONAL_MODEL
//         );
//     }
//     else if (model_ == CameraModel::FISHEYE)
//     {
//         K_ = cv::Mat::eye(3, 3, CV_64F);
//         // K_.at<double>(0,0) = 458.348;
//         // K_.at<double>(1,1) = 458.241;
//         // K_.at<double>(0,2) = 650.;
//         // K_.at<double>(1,2) = 531.;
//         D_ = cv::Mat::zeros(4, 1, CV_64F);
//         // D_.at<double>(0,0) = -0.22182901;
//         // D_.at<double>(1,0) =  0.17537102;
//         // D_.at<double>(2,0) =  0.00051886;
//         // D_.at<double>(3,0) = -0.00139967;

//         std::vector<cv::Mat> rvecs, tvecs;

//         int flags =
//             cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
//             //cv::fisheye::CALIB_CHECK_COND |
//             //cv::fisheye::CALIB_USE_INTRINSIC_GUESS;

//         reproj_error_ = cv::fisheye::calibrate(
//             object_points_,
//             image_points_,
//             image_size_,
//             K_,
//             D_,
//             rvecs,
//             tvecs,
//             flags
//         );
//     }

//     // --- Rapport de calibration ---
//     std::cout << "===== Camera Calibration Report =====" << std::endl;
//     std::cout << "Model: " << (model_ == CameraModel::PINHOLE ? "PINHOLE" : "FISHEYE") << std::endl;
//     std::cout << "Image size: " << image_size_.width << " x " << image_size_.height << std::endl;
//     std::cout << "Reprojection error: " << reproj_error_ << " pixels" << std::endl;
//     std::cout << "Intrinsic matrix (K): " << std::endl << K_ << std::endl;
//     std::cout << "Distortion coefficients (D): " << std::endl << D_ << std::endl;


//     return true;
// }

const cv::Mat& MonoCalibrator::cameraMatrix() const
{
    return K_;
}

const cv::Mat& MonoCalibrator::distCoeffs() const
{
    return D_;
}


