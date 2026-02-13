#include "aprilgrid_calibration_core/acalibrator.hpp"

class StereoCalibratorDS : public ACalibrator
{
public:
    StereoCalibratorDS(int image_width,
                     int image_height,
                     CameraModel model = CameraModel::DOUBLE_SPHERE)
            : ACalibrator(image_width, image_height, model) {};

    void addView(const std::vector<cv::Point2f>& pts2D_0,
                 const std::vector<cv::Point2f>& pts2D_1,
                 const std::vector<cv::Point3f>& pts3D);

    bool ready() const override;
    bool calibrate() override;

    std::optional<Extrinsics> getExtrinsics() const override {
        cv::Mat rvec = (cv::Mat_<double>(3,1) 
                        << RT_stereo_[0], RT_stereo_[1], RT_stereo_[2]);

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat t = (cv::Mat_<double>(3,1) 
                << RT_stereo_[3], RT_stereo_[4], RT_stereo_[5]);
        return Extrinsics{R, t};
    }

private:

    bool run_calibration(double* intrinsics0, double* intrinsics1,std::array<double,6>& RT_stereo, std::vector<std::array<double,6>>& RT_views);

    void initIntrinsics(const cv::Mat& K,  const cv::Mat& D, double intrinsics[6]);

    void createProblem(ceres::Problem& problem, ceres::ParameterBlockOrdering* ordering,
        double* intrinsics0,
        double* intrinsics1,
        std::array<double,6>& RT_stereo,
        std::vector<std::array<double,6>>& RT_views,
        bool use_soft_priors);

    bool addViewResiduals(ceres::Problem& problem,
            size_t view_id, double* intrinsics0, double* intrinsics1,
            std::array<double,6>& RT_stereo,
            std::array<double,6>& RT_view,
            bool use_soft_priors);

    void setParametersBounds(ceres::Problem& problem, double* intrinsics);

    ceres::Solver::Summary solveProblem(ceres::Problem& problem,
            ceres::ParameterBlockOrdering* ordering);

    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<cv::Point2f>> image_0_points_, image_1_points_;

    std::array<double,6> RT_stereo_;



};



// ============================================== Stereo ==============================================
// --- Double Sphere stereo Residual ---

struct ReprojectionErrorStereoDS {
    ReprojectionErrorStereoDS(const cv::Point3f& X,
                              const cv::Point2f& x0,
                              const cv::Point2f& x1)
        : X_(X), x0_(x0), x1_(x1) {}

    template <typename T>
    bool operator()(const T* const intrinsics0,
                    const T* const intrinsics1,
                    const T* const RT_board,
                    const T* const RT_cam1_cam0,
                    T* residuals) const
    {
        // cam0
        T Pw[3] = {T(X_.x), T(X_.y), T(X_.z)};
        T Pc0[3];
        ceres::AngleAxisRotatePoint(RT_board, Pw, Pc0);
        Pc0[0] += RT_board[3]; Pc0[1] += RT_board[4]; Pc0[2] += RT_board[5];

        T u0, v0;
        projectDoubleSphere(Pc0[0], Pc0[1], Pc0[2], intrinsics0, u0, v0);

        // cam1
        T Pc1[3];
        ceres::AngleAxisRotatePoint(RT_cam1_cam0, Pc0, Pc1);
        Pc1[0] += RT_cam1_cam0[3]; Pc1[1] += RT_cam1_cam0[4]; Pc1[2] += RT_cam1_cam0[5];

        T u1, v1;
        projectDoubleSphere(Pc1[0], Pc1[1], Pc1[2], intrinsics1, u1, v1);

        residuals[0] = u0 - T(x0_.x);
        residuals[1] = v0 - T(x0_.y);
        residuals[2] = u1 - T(x1_.x);
        residuals[3] = v1 - T(x1_.y);

        return true;
    }

    cv::Point3f X_;
    cv::Point2f x0_;
    cv::Point2f x1_;
};


// Ray stereo
struct ReprojectionErrorStereoDS_Ray {
    ReprojectionErrorStereoDS_Ray(const cv::Point3f& X,
                                  const cv::Point2f& x0,
                                  const cv::Point2f& x1)
        : X_(X), x0_(x0), x1_(x1) {}

    template <typename T>
    bool operator()(const T* const intrinsics0,
                    const T* const intrinsics1,
                    const T* const RT_board,
                    const T* const RT_cam1_cam0,
                    T* residuals) const
    {
        T ray0[3], ray1[3];
        unprojectDoubleSphere(T(x0_.x), T(x0_.y), intrinsics0, ray0);
        unprojectDoubleSphere(T(x1_.x), T(x1_.y), intrinsics1, ray1);

        T Pw[3] = {T(X_.x), T(X_.y), T(X_.z)};
        T Pc0[3];
        ceres::AngleAxisRotatePoint(RT_board, Pw, Pc0);
        Pc0[0] += RT_board[3]; Pc0[1] += RT_board[4]; Pc0[2] += RT_board[5];

        T norm0 = sqrt(Pc0[0]*Pc0[0]+Pc0[1]*Pc0[1]+Pc0[2]*Pc0[2]);
        if(norm0<T(1e-9)) norm0=T(1.0);
        T ray0_pred[3] = {Pc0[0]/norm0,Pc0[1]/norm0,Pc0[2]/norm0};

        residuals[0] = ray0_pred[1]*ray0[2] - ray0_pred[2]*ray0[1];
        residuals[1] = ray0_pred[2]*ray0[0] - ray0_pred[0]*ray0[2];
        residuals[2] = ray0_pred[0]*ray0[1] - ray0_pred[1]*ray0[0];

        T Pc1[3];
        ceres::AngleAxisRotatePoint(RT_cam1_cam0, Pc0, Pc1);
        Pc1[0]+=RT_cam1_cam0[3]; Pc1[1]+=RT_cam1_cam0[4]; Pc1[2]+=RT_cam1_cam0[5];

        T norm1 = sqrt(Pc1[0]*Pc1[0]+Pc1[1]*Pc1[1]+Pc1[2]*Pc1[2]);
        if(norm1<T(1e-9)) norm1=T(1.0);
        T ray1_pred[3] = {Pc1[0]/norm1,Pc1[1]/norm1,Pc1[2]/norm1};

        residuals[3] = ray1_pred[1]*ray1[2] - ray1_pred[2]*ray1[1];
        residuals[4] = ray1_pred[2]*ray1[0] - ray1_pred[0]*ray1[2];
        residuals[5] = ray1_pred[0]*ray1[1] - ray1_pred[1]*ray1[0];

        return true;
    }

    cv::Point3f X_;
    cv::Point2f x0_;
    cv::Point2f x1_;
};


// --- Contrainte épipolaire rayon de vue ---
struct ReprojectionErrorStereoDS_Ray_Epipolar {
    ReprojectionErrorStereoDS_Ray_Epipolar(
        const cv::Point3f& X,
        const cv::Point2f& x0,
        const cv::Point2f& x1)
        : X_(X), x0_(x0), x1_(x1) {}

    template <typename T>
    bool operator()(const T* const intrinsics0,
                    const T* const intrinsics1,
                    const T* const RT_board,
                    const T* const RT_cam1_cam0,
                    T* residuals) const
    {
        // ----- Point dans cam0 -----
        T Pw[3] = {T(X_.x), T(X_.y), T(X_.z)};
        T Pc0[3];
        ceres::AngleAxisRotatePoint(RT_board, Pw, Pc0);
        Pc0[0] += RT_board[3];
        Pc0[1] += RT_board[4];
        Pc0[2] += RT_board[5];

        // Projection en rayons
        T ray0[3];
        unprojectDoubleSphere(T(x0_.x), T(x0_.y), intrinsics0, ray0);

        // ----- Point dans cam1 -----
        T Pc1[3];
        ceres::AngleAxisRotatePoint(RT_cam1_cam0, Pc0, Pc1);
        Pc1[0] += RT_cam1_cam0[3];
        Pc1[1] += RT_cam1_cam0[4];
        Pc1[2] += RT_cam1_cam0[5];

        T ray1[3];
        unprojectDoubleSphere(T(x1_.x), T(x1_.y), intrinsics1, ray1);


        // ----- Résidu épipolaire (Sampson / linéaire) -----
        // ray0 et ray1 doivent être des rayons unitaires
        T t[3] = {RT_cam1_cam0[3], RT_cam1_cam0[4], RT_cam1_cam0[5]}; // translation
        T r[3] = {RT_cam1_cam0[0], RT_cam1_cam0[1], RT_cam1_cam0[2]}; // angle-axis

        // rotation de cam0 vers cam1
        T R[9];
        ceres::AngleAxisToRotationMatrix(r, R);

        // épipole : e = t x R*ray0
        T Rray0[3];
        Rray0[0] = R[0]*ray0[0] + R[1]*ray0[1] + R[2]*ray0[2];
        Rray0[1] = R[3]*ray0[0] + R[4]*ray0[1] + R[5]*ray0[2];
        Rray0[2] = R[6]*ray0[0] + R[7]*ray0[1] + R[8]*ray0[2];

        T ex = t[1]*Rray0[2] - t[2]*Rray0[1];
        T ey = t[2]*Rray0[0] - t[0]*Rray0[2];
        T ez = t[0]*Rray0[1] - t[1]*Rray0[0];

        // épipolaire : dot(ray1, e)
        residuals[0] = ray1[0]*ex + ray1[1]*ey + ray1[2]*ez;

        return true;
    }

    cv::Point3f X_;
    cv::Point2f x0_;
    cv::Point2f x1_;
};