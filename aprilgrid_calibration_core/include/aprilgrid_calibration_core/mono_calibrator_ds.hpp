#include "aprilgrid_calibration_core/acalibrator.hpp"

class MonoCalibratorDS : public ACalibrator
{
public:
    MonoCalibratorDS(int image_width,
                     int image_height,
                     CameraModel model = CameraModel::DOUBLE_SPHERE)
            : ACalibrator(image_width, image_height, model) {};

    void addView(const std::vector<cv::Point2f>& pts2D,
                 const std::vector<cv::Point3f>& pts3D);

    bool ready() const override;
    bool calibrate() override;    

    std::optional<Extrinsics> getExtrinsics() const override {
        return std::nullopt; 
    }

private:

    bool run_calibration(double* intrinsics, std::vector<std::array<double,6>>& rts);

    void createProblem(ceres::Problem& problem, ceres::ParameterBlockOrdering* ordering,
            double* intrinsics,
            std::vector<std::array<double,6>>& RT_views,
            bool use_soft_priors);

    bool addViewResiduals(ceres::Problem& problem,
            size_t view_id, double* intrinsics,
            std::array<double,6>& RT_view,
            bool use_soft_priors);

    void setParametersBounds(ceres::Problem& problem, double* intrinsics);

    ceres::Solver::Summary solveProblem(ceres::Problem& problem,
            ceres::ParameterBlockOrdering* ordering);

    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<cv::Point2f>> image_points_;

};






// ============================================== Mono ==============================================
// --- Double Sphere Mono Residual ---

struct ReprojectionErrorDS {
    ReprojectionErrorDS(const cv::Point3f& X, const cv::Point2f& x)
        : X_(X), x_(x) {}

    template <typename T>
    bool operator()(const T* const intrinsics,
                    const T* const rt,
                    T* residuals) const
    {
        T Pc[3] = {T(X_.x), T(X_.y), T(X_.z)};
        ceres::AngleAxisRotatePoint(rt, Pc, Pc);
        Pc[0] += rt[3]; Pc[1] += rt[4]; Pc[2] += rt[5];

        T u, v;
        projectDoubleSphere(Pc[0], Pc[1], Pc[2], intrinsics, u, v);

        residuals[0] = u - T(x_.x);
        residuals[1] = v - T(x_.y);
        return true;
    }

    cv::Point3f X_;
    cv::Point2f x_;
};


struct ReprojectionErrorDS_Ray {
    ReprojectionErrorDS_Ray(const cv::Point3f& X, const cv::Point2f& x) : X_(X), x_(x) {}

    template <typename T>
    bool operator()(const T* const intrinsics,
                    const T* const rt,
                    T* residuals) const
    {
        T ray_obs[3];
        unprojectDoubleSphere(T(x_.x), T(x_.y), intrinsics, ray_obs);

        T Pw[3] = {T(X_.x), T(X_.y), T(X_.z)};
        T Pc[3];
        ceres::AngleAxisRotatePoint(rt, Pw, Pc);
        Pc[0] += rt[3]; Pc[1] += rt[4]; Pc[2] += rt[5];

        T norm = sqrt(Pc[0]*Pc[0] + Pc[1]*Pc[1] + Pc[2]*Pc[2]);
        if (norm < T(1e-9)) norm = T(1.0);

        T ray_pred[3] = { Pc[0]/norm, Pc[1]/norm, Pc[2]/norm };

        residuals[0] = ray_pred[1]*ray_obs[2] - ray_pred[2]*ray_obs[1];
        residuals[1] = ray_pred[2]*ray_obs[0] - ray_pred[0]*ray_obs[2];
        residuals[2] = ray_pred[0]*ray_obs[1] - ray_pred[1]*ray_obs[0];

        return true;
    }

    cv::Point3f X_;
    cv::Point2f x_;
};