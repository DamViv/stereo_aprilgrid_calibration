#include "../include/aprilgrid_calibration_core/stereo_calibrator_ds.hpp"
#include "aprilgrid_calibration_core/SE3_tool.hpp"


void StereoCalibratorDS::addView(const std::vector<cv::Point2f>& pts2D_0,
                                 const std::vector<cv::Point2f>& pts2D_1,
                                 const std::vector<cv::Point3f>& pts3D)
{
    if (pts2D_0.size() != pts3D.size() || pts2D_0.size() != pts2D_1.size())
        return;

    if (pts2D_0.size() < 10 || pts2D_1.size() < 10)
        return;

    image_0_points_.push_back(pts2D_0);
    image_1_points_.push_back(pts2D_1);
    object_points_.push_back(pts3D);
}



bool StereoCalibratorDS::ready() const
{
    return image_0_points_.size() >= 10 && image_1_points_.size() >= 10;
}



void StereoCalibratorDS::initIntrinsics(const cv::Mat& K,  const cv::Mat& D, double intrinsics[6]){
    intrinsics[0] = K.at<double>(0,0);
    intrinsics[1] = K.at<double>(1,1);
    intrinsics[2] = K.at<double>(0,2);
    intrinsics[3] = K.at<double>(1,2);
    intrinsics[4] = D.at<double>(0);
    intrinsics[5] = D.at<double>(1);   
}



bool StereoCalibratorDS::calibrate() {
    if (!ready()) return false;

    // Intrinsics init            
    double intrinsics0[6], intrinsics1[6];    
    initIntrinsics(K_.at(0), D_.at(0), intrinsics0);        
    initIntrinsics(K_.at(1), D_.at(1), intrinsics1);    

    // Initialisation des poses caméra (RT) par vue
    std::vector<std::array<double,6>> RT_mire_views(object_points_.size(), {0,0,0,0,0,0});
    std::array<double,6> RT_stereo = {0,0,0,0.2,0,0}; // initialement alignées
        
    // PnP init of baseline
    std::vector<cv::Mat> stereo_estimates;
    cv::Mat D0_cv = cv::Mat::zeros(4,1,CV_64F);
    cv::Mat D1_cv = cv::Mat::zeros(4,1,CV_64F);    

    for (size_t i = 0; i < object_points_.size(); i++) {
        cv::Mat rvecL, tvecL, rvecR, tvecR;
        cv::solvePnP(object_points_[i], image_0_points_[i], K_.at(0), D0_cv, rvecL, tvecL);
        cv::solvePnP(object_points_[i], image_1_points_[i], K_.at(1), D1_cv, rvecR, tvecR);
        cv::Mat RL, RR;
        cv::Rodrigues(rvecL, RL);
        cv::Rodrigues(rvecR, RR);
        cv::Mat R_stereo = RR * RL.t();
        cv::Mat t_stereo = tvecR - R_stereo * tvecL;
        cv::Mat RT(4,4,CV_64F);
        RT.setTo(0);
        R_stereo.copyTo(RT(cv::Rect(0,0,3,3)));
        t_stereo.copyTo(RT(cv::Rect(3,0,1,3)));
        RT.at<double>(3,3) = 1;
        stereo_estimates.push_back(RT);
    }

    // median rotation + translation (Lie algebra averaging)
    cv::Mat RT_init = medianSE3(stereo_estimates);

    // Convert RT_init -> array<double,6> pour Ceres
    cv::Mat R = RT_init(cv::Rect(0,0,3,3));
    cv::Mat t = RT_init(cv::Rect(3,0,1,3));

    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    RT_stereo = {
        rvec.at<double>(0),
        rvec.at<double>(1),
        rvec.at<double>(2),
        t.at<double>(0),
        t.at<double>(1),
        t.at<double>(2)
    };

    return run_calibration(intrinsics0, intrinsics1, RT_stereo, RT_mire_views);
}



bool StereoCalibratorDS::run_calibration(double* intrinsics0, double* intrinsics1,std::array<double,6>& RT_stereo, std::vector<std::array<double,6>>& RT_views){


    // ---------------------------
    // STAGE 1 — poses only
    // ---------------------------
    std::cout << "\n=== Stage 1: optimize poses only ===\n";
    {
        ceres::Problem problem;
        auto ordering = new ceres::ParameterBlockOrdering;
        createProblem(problem, ordering, intrinsics0, intrinsics1, RT_stereo, RT_views, false);

        problem.SetParameterBlockConstant(intrinsics0);
        problem.SetParameterBlockConstant(intrinsics1);
        problem.SetParameterBlockConstant(RT_stereo.data());

        setParametersBounds(problem, intrinsics0);
        setParametersBounds(problem, intrinsics1);
        solveProblem(problem, ordering);

        // // ---------------------------
        // // PRINT RESULTS
        // // ---------------------------
        // std::cout << "\n=== Intrinsics Camera 0 ===\n";
        // displayIntrinsics(intrinsics0, model_);
        // std::cout << "\n=== Intrinsics Camera 1 ===\n";
        // displayIntrinsics(intrinsics1, model_);
        // std::cout << "\n=== Stereo Extrinsics ===\n";
        // displayExtrinsics(RT_stereo);

    }

    // ---------------------------
    // STAGE 2 — poses + stereo
    // ---------------------------
    std::cout << "\n=== Stage 2: optimize stereo ===\n";
    {
        ceres::Problem problem;
        auto ordering = new ceres::ParameterBlockOrdering;
        createProblem(problem, ordering, intrinsics0, intrinsics1, RT_stereo, RT_views, false);

        problem.SetParameterBlockConstant(intrinsics0);
        problem.SetParameterBlockConstant(intrinsics1);

        setParametersBounds(problem, intrinsics0);
        setParametersBounds(problem, intrinsics1);
        solveProblem(problem, ordering);

        // // ---------------------------
        // // PRINT RESULTS
        // // ---------------------------
        // std::cout << "\n=== Intrinsics Camera 0 ===\n";
        // displayIntrinsics(intrinsics0, model_);
        // std::cout << "\n=== Intrinsics Camera 1 ===\n";
        // displayIntrinsics(intrinsics1, model_);
        // std::cout << "\n=== Stereo Extrinsics ===\n";
        // displayExtrinsics(RT_stereo);        
    }

    // ---------------------------
    // STAGE 3 — unlock focal group
    // ---------------------------
    std::cout << "\n=== Stage 3: unlock fx fy ===\n";
    {
        ceres::Problem problem;
        auto ordering = new ceres::ParameterBlockOrdering;
        createProblem(problem, ordering, intrinsics0, intrinsics1, RT_stereo, RT_views, true);

        std::vector<int> fixed = {2,3,4,5}; // lock cx cy xi alpha
        problem.SetManifold(intrinsics0, new ceres::SubsetManifold(6, fixed));
        problem.SetManifold(intrinsics1, new ceres::SubsetManifold(6, fixed));

        setParametersBounds(problem, intrinsics0);
        setParametersBounds(problem, intrinsics1);
        solveProblem(problem, ordering);

        // // ---------------------------
        // // PRINT RESULTS
        // // ---------------------------
        // std::cout << "\n=== Intrinsics Camera 0 ===\n";
        // displayIntrinsics(intrinsics0, model_);
        // std::cout << "\n=== Intrinsics Camera 1 ===\n";
        // displayIntrinsics(intrinsics1, model_);
        // std::cout << "\n=== Stereo Extrinsics ===\n";
        // displayExtrinsics(RT_stereo);        
    }

    // ---------------------------
    // STAGE 4 — unlock center + distortion (GROUPED)
    // ---------------------------
    std::cout << "\n=== Stage 4: unlock all intrinsics ===\n";
    {
        ceres::Problem problem;
        auto ordering = new ceres::ParameterBlockOrdering;
        createProblem(problem, ordering, intrinsics0, intrinsics1, RT_stereo, RT_views, true);

        setParametersBounds(problem, intrinsics0);
        setParametersBounds(problem, intrinsics1);
        solveProblem(problem, ordering);

        // // ---------------------------
        // // PRINT RESULTS
        // // ---------------------------
        // std::cout << "\n=== Intrinsics Camera 0 ===\n";
        // displayIntrinsics(intrinsics0, model_);
        // std::cout << "\n=== Intrinsics Camera 1 ===\n";
        // displayIntrinsics(intrinsics1, model_);
        // std::cout << "\n=== Stereo Extrinsics ===\n";
        // displayExtrinsics(RT_stereo);        
    }

    // ---------------------------
    // FINAL GLOBAL PASS
    // ---------------------------
    std::cout << "\n=== FINAL GLOBAL PASS ===\n";

    ceres::Problem problem_final;
    auto ordering_final = new ceres::ParameterBlockOrdering;
    createProblem(problem_final, ordering_final, intrinsics0, intrinsics1, RT_stereo, RT_views, true);

    setParametersBounds(problem_final, intrinsics0);
    setParametersBounds(problem_final, intrinsics1);
    ceres::Solver::Summary summary_final = solveProblem(problem_final, ordering_final);


    // ---------------------------
    // COVARIANCE COMPUTATION
    // ---------------------------
    ceres::Covariance::Options cov_options;
    cov_options.null_space_rank = -1;
    cov_options.algorithm_type = ceres::DENSE_SVD;
    ceres::Covariance covariance(cov_options);  
    
    std::vector<std::pair<const double*, const double*>> blocks;
    blocks.emplace_back(intrinsics0, intrinsics0);
    blocks.emplace_back(intrinsics1, intrinsics1);
    covariance.Compute(blocks, &problem_final);

    double cov_intr0[36], cov_intr1[36], cov_stereo[36];
    covariance.GetCovarianceBlock(intrinsics0, intrinsics0, cov_intr0);
    covariance.GetCovarianceBlock(intrinsics1, intrinsics1, cov_intr1);
    covariance.GetCovarianceBlock(RT_stereo.data(), RT_stereo.data(), cov_stereo);


    // ---------------------------
    // PRINT RESULTS
    // ---------------------------
    std::cout << "\n=== Intrinsics Camera 0 ===\n";
    displayIntrinsics(intrinsics0, model_, cov_intr0);
    std::cout << "\n=== Intrinsics Camera 1 ===\n";
    displayIntrinsics(intrinsics1, model_, cov_intr1);
    std::cout << "\n=== Stereo Extrinsics ===\n";
    displayExtrinsics(RT_stereo, cov_stereo);
    // std::cout << "\n=== Intrinsics Camera 0 ===\n";
    // displayIntrinsics(intrinsics0, model_);
    // std::cout << "\n=== Intrinsics Camera 1 ===\n";
    // displayIntrinsics(intrinsics1, model_);
    // std::cout << "\n=== Stereo Extrinsics ===\n";
    // displayExtrinsics(RT_stereo);
    std::cout << "\n=== Final RMSE ===\n";
    double final_rmse = computeRMSE(summary_final);
    std::cout << final_rmse << std::endl;


    // ---------------------------
    // STORE RESULTS
    // ---------------------------
    cv::Mat K, D;
    K = (cv::Mat_<double>(3,3) << intrinsics0[0], 0, intrinsics0[2],
                                    0, intrinsics0[1], intrinsics0[3],
                                    0, 0, 1);
    K_.at(0) = K.clone();
    D = (cv::Mat_<double>(6,1) << intrinsics0[4], intrinsics0[5], 0, 0, 0, 0);
    D_.at(0) = D.clone();
    
    K = (cv::Mat_<double>(3,3) << intrinsics1[0], 0, intrinsics1[2],
                                    0, intrinsics1[1], intrinsics1[3],
                                    0, 0, 1);
    K_.at(1) = K.clone();
    D = (cv::Mat_<double>(6,1) << intrinsics1[4], intrinsics1[5], 0, 0, 0, 0);
    D_.at(1) = D.clone();
    RT_stereo_ = RT_stereo;

    reproj_error_ = final_rmse;

    return summary_final.termination_type == ceres::CONVERGENCE;
}



void StereoCalibratorDS::createProblem(ceres::Problem& problem, ceres::ParameterBlockOrdering* ordering,
    double* intrinsics0,
    double* intrinsics1,
    std::array<double,6>& RT_stereo,
    std::vector<std::array<double,6>>& RT_views,
    bool use_soft_priors){
    
    // ---- Global parameters
    problem.AddParameterBlock(intrinsics0, 6);
    problem.AddParameterBlock(intrinsics1, 6);
    problem.AddParameterBlock(RT_stereo.data(), 6);

    // ---- Schur ordering    
    ordering->AddElementToGroup(intrinsics0, 1);
    ordering->AddElementToGroup(intrinsics1, 1);
    ordering->AddElementToGroup(RT_stereo.data(), 1);
    // Group 0 = Schur (poses)
    for (auto& rt : RT_views) {
        ordering->AddElementToGroup(rt.data(), 0);
    }


    size_t n_views = object_points_.size();    
    for (size_t i=0; i<n_views; ++i) {
    
        if(addViewResiduals(problem, i,
                        intrinsics0,
                        intrinsics1,
                        RT_stereo,
                        RT_views[i],
                        use_soft_priors))
            ordering->AddElementToGroup(RT_views[i].data(), 0);
    }
}



bool StereoCalibratorDS::addViewResiduals(ceres::Problem& problem,
        size_t view_id, double* intrinsics0, double* intrinsics1,
        std::array<double,6>& RT_stereo,
        std::array<double,6>& RT_view,
        bool use_soft_priors){

    const auto& pts3D = object_points_[view_id];
    const auto& pts0  = image_0_points_[view_id];
    const auto& pts1  = image_1_points_[view_id];
    if (pts3D.size() != pts0.size() || pts0.size() != pts1.size()) {
        std::cerr << "[StereoCalib] size mismatch view" << "\n";
        return false;
    }

    for (size_t j = 0; j < pts3D.size(); ++j) {

        ceres::CostFunction* cost_pixel =
            new ceres::AutoDiffCostFunction<ReprojectionErrorStereoDS, 4, 6, 6, 6, 6>(
                new ReprojectionErrorStereoDS(pts3D[j], pts0[j], pts1[j]));

        problem.AddResidualBlock(cost_pixel,
                                 new ceres::HuberLoss(1.0),
                                 intrinsics0, intrinsics1,
                                 RT_view.data(),
                                 RT_stereo.data());

        if(use_soft_priors){
            // ---- Soft priors init
            double fx0_init = intrinsics0[0];
            double fx1_init = intrinsics1[0];
            double xi0_init = intrinsics0[4];
            double xi1_init = intrinsics1[4];
            double baseline_init = RT_stereo[3];

            // ----------------------------------
            // SubsetManifolds : bloquer les autres paramètres
            // ----------------------------------
            // intrinsics0 : bloc de 6 -> bloquer fy,cx,cy,alpha
            std::vector<int> fixed_intrinsics0 = {1,2,3,5}; 
            problem.SetManifold(intrinsics0, new ceres::SubsetManifold(6, fixed_intrinsics0));

            // intrinsics1 : bloc de 6 -> bloquer fy,cx,cy,alpha
            std::vector<int> fixed_intrinsics1 = {1,2,3,5};
            problem.SetManifold(intrinsics1, new ceres::SubsetManifold(6, fixed_intrinsics1));

            // ----------------------------------
            // Add priors on fx (index=0)
            // ----------------------------------
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(fx0_init, fx0_init*0.03, 0)), // fx0
                nullptr, intrinsics0);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(fx1_init, fx1_init*0.03, 0)), // fx1
                nullptr, intrinsics1);

            // ----------------------------------
            // Add priors on xi (index=4)
            // ----------------------------------
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(xi0_init, 0.02, 4)), // xi0
                nullptr, intrinsics0);

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(xi1_init, 0.02, 4)), // xi1
                nullptr, intrinsics1);

            // ----------------------------------
            // Add prior on stereo baseline (translation x, index=3)
            // ----------------------------------
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PriorResidual,1,6>(
                    new PriorResidual(baseline_init, baseline_init*0.05, 3)), // t_x
                nullptr, RT_stereo.data());
        }
    }
    return true;
}



void StereoCalibratorDS::setParametersBounds(ceres::Problem& problem, double* intrinsics){
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



ceres::Solver::Summary StereoCalibratorDS::solveProblem(ceres::Problem& problem,
                                      ceres::ParameterBlockOrdering* ordering)
{
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.linear_solver_ordering.reset(ordering);
    options.use_explicit_schur_complement      = true;
    
    options.max_num_iterations                 = 100;
    options.minimizer_progress_to_stdout       = false;    
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-6;
    options.num_threads                        = 8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return summary;
}