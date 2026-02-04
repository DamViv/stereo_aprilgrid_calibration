#ifndef DOUBLE_SPHERE_HPP
#define DOUBLE_SPHERE_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// ==================================== Reprojection double sphere ======================================

template <typename T>
inline bool projectDoubleSphere(
    const T& x, const T& y, const T& z,
    const T* intrinsics,   // fx fy cx cy xi alpha
    T& u, T& v)
{
    const T& fx = intrinsics[0];
    const T& fy = intrinsics[1];
    const T& cx = intrinsics[2];
    const T& cy = intrinsics[3];
    const T& xi = intrinsics[4];
    const T& alpha = intrinsics[5];

    const T eps = T(1e-8);

    T d1 = sqrt(x*x + y*y + z*z);
    if (d1 < eps) return false;

    T z1 = z + xi * d1;
    if (z1 < eps) return false;

    T d2 = sqrt(x*x + y*y + z1*z1);
    T denom = alpha * d2 + (T(1.0) - alpha) * z1;

    if (denom < eps) return false;

    T mx = x / denom;
    T my = y / denom;

    u = fx * mx + cx;
    v = fy * my + cy;

    return true;
}


// ============================================== Mono ==============================================
// --- Double Sphere Mono Residual ---

struct ReprojectionErrorDS {
    ReprojectionErrorDS(const cv::Point3f& X, const cv::Point2f& x) : X_(X), x_(x) {}

    template <typename T>
    bool operator()(const T* const intrinsics,  // fx,fy,cx,cy,xi,alpha
                    const T* const rt,          // rx,ry,rz,tx,ty,tz
                    T* residuals) const 
    {

        // Point caméra
        T p[3] = {T(X_.x), T(X_.y), T(X_.z)};
        T pc0[3];
        ceres::AngleAxisRotatePoint(rt, p, pc0);
        pc0[0] += rt[3];
        pc0[1] += rt[4];
        pc0[2] += rt[5];

        // Projection
        T u, v;
        if (!projectDoubleSphere(pc0[0], pc0[1], pc0[2], intrinsics, u, v)){
            // Important : pénaliser doucement si projection invalide (??? ou ignore?)
            residuals[0] = T(10);
            residuals[1] = T(10);
            return true;
        }        
        
        // Résiduals
        residuals[0] = u - T(x_.x);
        residuals[1] = v - T(x_.y);

        return true;
    }

    cv::Point3f X_;
    cv::Point2f x_;
};



// ============================================== Stereo ==============================================
// --- Double Sphere Mono Residual ---

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
        // Point monde → cam0
        T Pw[3] = {T(X_.x), T(X_.y), T(X_.z)};
        T Pc0[3];
        ceres::AngleAxisRotatePoint(RT_board, Pw, Pc0);
        Pc0[0] += RT_board[3];
        Pc0[1] += RT_board[4];
        Pc0[2] += RT_board[5];

        // Projection cam0
        T u0, v0;
        if (!projectDoubleSphere(Pc0[0], Pc0[1], Pc0[2], intrinsics0, u0, v0)){
            residuals[0] = T(10);
            residuals[1] = T(10);
            residuals[2] = T(10);
            residuals[3] = T(10);
            return true;
        }

        // cam0 → cam1
        T Pc1[3];
        ceres::AngleAxisRotatePoint(RT_cam1_cam0, Pc0, Pc1);
        Pc1[0] += RT_cam1_cam0[3];
        Pc1[1] += RT_cam1_cam0[4];
        Pc1[2] += RT_cam1_cam0[5];

        // Projection cam1
        T u1, v1;
        if (!projectDoubleSphere(Pc1[0], Pc1[1], Pc1[2], intrinsics1, u1, v1)){
           residuals[0] = T(10);
            residuals[1] = T(10);
            residuals[2] = T(10);
            residuals[3] = T(10);
            return true;
        }

        // Résidus
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



// Projection equirectangular
cv::Mat generateEquirectangular(const cv::Mat& img, const cv::Mat& K, const cv::Mat& D, int width, int height);

cv::Mat undistortDoubleSphere(const cv::Mat& img, const cv::Mat& K, const cv::Mat& D);


#endif //DOUBLE_SPHERE_HPP