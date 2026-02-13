#include "../include/aprilgrid_calibration_core/mono_calibrator_ds.hpp"


void MonoCalibratorDS::addView(const std::vector<cv::Point2f>& pts2D,
                             const std::vector<cv::Point3f>& pts3D)
{
    if (pts2D.size() != pts3D.size())
        return;

    if (pts2D.size() < 10)
        return;

    image_points_.push_back(pts2D);
    object_points_.push_back(pts3D);
}

bool MonoCalibratorDS::ready() const
{
    return image_points_.size() >= 10;
}


bool MonoCalibratorDS::calibrate() {
    if (!ready()) return false;


    // Intrinsec init
    double intrinsics[6] = {(double)image_size_.width/3., (double)image_size_.height/3., (double)image_size_.width/2., (double)image_size_.height/2., 0.0, 0.5};


    // Initialisation des poses caméra (RT) par vue
    std::vector<std::array<double,6>> RT_mire_views(object_points_.size(), {0,0,0,0,0,0});

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

        RT_mire_views[i][0] = rvec.at<double>(0);
        RT_mire_views[i][1] = rvec.at<double>(1);
        RT_mire_views[i][2] = rvec.at<double>(2);
        RT_mire_views[i][3] = tvec.at<double>(0);
        RT_mire_views[i][4] = tvec.at<double>(1);
        RT_mire_views[i][5] = tvec.at<double>(2);
    }

    return run_calibration(intrinsics, RT_mire_views);
}


bool MonoCalibratorDS::run_calibration(double* intrinsics, std::vector<std::array<double,6>>& RT_views){


    ceres::Problem problem;
    auto ordering = new ceres::ParameterBlockOrdering;
    createProblem(problem, ordering, intrinsics, RT_views, false);

    setParametersBounds(problem, intrinsics);

    ceres::Solver::Summary summary = solveProblem(problem, ordering);
    double rms = computeRMSE(summary);  

    std::cout << "Reprojection RMSE = " << rms << " pixels" << std::endl;

    std::cout << "\n=== Intrinsics Camera ===\n";
    displayIntrinsics(intrinsics, model_);

    // // ---------------------------
    // // COVARIANCE COMPUTATION
    // // ---------------------------
    // ceres::Covariance::Options cov_options;
    // cov_options.null_space_rank = -1;
    // cov_options.algorithm_type = ceres::DENSE_SVD;
    // //cov_options.algorithm_type = ceres::SPARSE_QR;
    // ceres::Covariance covariance(cov_options);  
    
    // std::vector<std::pair<const double*, const double*>> blocks;
    // blocks.emplace_back(intrinsics, intrinsics);
    // covariance.Compute(blocks, &problem);

    // double cov_intr[36];
    // covariance.GetCovarianceBlock(intrinsics, intrinsics, cov_intr);
    // std::cout << "\n=== Intrinsics Camera with cov ===\n";
    // displayIntrinsics(intrinsics, model_, cov_intr);
    std::cout << "\n=== Intrinsics Camera ===\n";
    displayIntrinsics(intrinsics, model_);

    // Stockage des résultats de calibration
    cv::Mat K = (cv::Mat_<double>(3,3) << intrinsics[0], 0, intrinsics[2],
                                    0, intrinsics[1], intrinsics[3],
                                    0, 0, 1);
    K_.at(0) = K.clone();

    cv::Mat D = (cv::Mat_<double>(6,1) << intrinsics[4], intrinsics[5], 0, 0, 0, 0);
    D_.at(0) = D.clone();
    
    reproj_error_ = rms;

    return summary.termination_type == ceres::CONVERGENCE;    
    
}


void MonoCalibratorDS::createProblem(ceres::Problem& problem, ceres::ParameterBlockOrdering* ordering,
    double* intrinsics,
    std::vector<std::array<double,6>>& RT_views,
    bool use_soft_priors){
    
    // ---- Global parameters
    problem.AddParameterBlock(intrinsics, 6);        

    // ---- Schur ordering    
    // Groupe 1 = global parameters
    ordering->AddElementToGroup(intrinsics, 1);        

    // Add residuals
    size_t n_views = object_points_.size();    
    for (size_t i=0; i<n_views; ++i) {
        if(addViewResiduals(problem, i,
                        intrinsics,                        
                        RT_views[i],
                        use_soft_priors))      
            ordering->AddElementToGroup(RT_views[i].data(), 0);     
    }
}

bool MonoCalibratorDS::addViewResiduals(ceres::Problem& problem,
        size_t view_id, double* intrinsics,
        std::array<double,6>& RT_view,
        bool use_soft_priors){

    const auto& pts3D = object_points_[view_id];
    const auto& pts0  = image_points_[view_id];
    if (pts3D.size() != pts0.size()) {
        std::cerr << "[MonoCalibDS] size mismatch view" << "\n";
        return false;
    }

    for (size_t j = 0; j < pts3D.size(); ++j) {

        ceres::CostFunction* cost_pixel =
            new ceres::AutoDiffCostFunction<ReprojectionErrorDS, 2, 6, 6>(
                new ReprojectionErrorDS(pts3D[j], pts0[j]));

        problem.AddResidualBlock(cost_pixel,
                                 new ceres::HuberLoss(1.0),
                                 intrinsics,
                                 RT_view.data());

        if(use_soft_priors){
            // ---- Soft priors init
            double fx0_init = intrinsics[0];
            double xi0_init = intrinsics[4];            
            
            // ----------------------------------
            // Add priors on fx (index=0)
            // ----------------------------------
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(fx0_init, fx0_init*0.05, 0)), // fx0
                nullptr, intrinsics);

            // ----------------------------------
            // Add priors on xi (index=4)
            // ----------------------------------
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(xi0_init, xi0_init*0.05, 4)), // xi0
                nullptr, intrinsics);
        }
    }
    return true;
}

void MonoCalibratorDS::setParametersBounds(ceres::Problem& problem, double* intrinsics){
    // Limit the parameters
    // fx, fy
    problem.SetParameterLowerBound(intrinsics, 0, 1.0);
    problem.SetParameterUpperBound(intrinsics, 0, image_size_.width/2.0);
    problem.SetParameterLowerBound(intrinsics, 1, 1.0);
    problem.SetParameterUpperBound(intrinsics, 1, image_size_.height/2.0);
    // cx, cy (centre)
    problem.SetParameterLowerBound(intrinsics, 2, image_size_.width/4.0);
    problem.SetParameterUpperBound(intrinsics, 2, 3.0*image_size_.width/4.0);
    problem.SetParameterLowerBound(intrinsics, 3, image_size_.height/4.0);
    problem.SetParameterUpperBound(intrinsics, 3, 3.0*image_size_.height/4.0);
    // xi, alpha
    problem.SetParameterLowerBound(intrinsics, 4, -0.1);
    problem.SetParameterUpperBound(intrinsics, 4, 2.0);
    problem.SetParameterLowerBound(intrinsics, 5, 0.0);
    problem.SetParameterUpperBound(intrinsics, 5, 1.0);
}


ceres::Solver::Summary MonoCalibratorDS::solveProblem(ceres::Problem& problem,
                                      ceres::ParameterBlockOrdering* ordering)
{
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.linear_solver_ordering.reset(ordering);
    options.use_explicit_schur_complement      = true;
    
    options.max_num_iterations                 = 500;
    options.minimizer_progress_to_stdout       = false;    
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-6;
    options.num_threads                        = 8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return summary;
}