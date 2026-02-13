#ifndef CAMERA_MODEL_HPP
#define CAMERA_MODEL_HPP

#include <cmath>


enum class CameraModel
{
    PINHOLE,
    FISHEYE,
    DOUBLE_SPHERE
};





// ==================================== Reprojection double sphere ======================================

template <typename T>
inline void projectDoubleSphere(
    const T& x, const T& y, const T& z,
    const T* intrinsics, 
    T& u, T& v)
{
    const T& fx = intrinsics[0];
    const T& fy = intrinsics[1];
    const T& cx = intrinsics[2];
    const T& cy = intrinsics[3];
    const T& xi = intrinsics[4];
    const T& alpha = intrinsics[5];

    const T eps = T(1e-8);

    // distance au centre
    T d1 = sqrt(x*x + y*y + z*z + eps);

    // z1 = z + xi*d1
    T z1 = z + xi * d1;
    if (z1 < eps) z1 = eps;

    // distance modifiée
    T d2 = sqrt(x*x + y*y + z1*z1 + eps);

    // dénominateur du modèle
    T denom = alpha * d2 + (T(1.0) - alpha) * z1;
    if (denom < eps) denom = eps;

    // projection pixel
    u = fx * x / denom + cx;
    v = fy * y / denom + cy;
}

template <typename T>
inline void unprojectDoubleSphere(
    const T& u, const T& v,
    const T* intrinsics, 
    T ray[3])
{
    const T& fx = intrinsics[0];
    const T& fy = intrinsics[1];
    const T& cx = intrinsics[2];
    const T& cy = intrinsics[3];
    const T& xi = intrinsics[4];
    const T& alpha = intrinsics[5];

    const T eps = T(1e-8);

    // coordonnées normalisées
    T mx = (u - cx) / fx;
    T my = (v - cy) / fy;
    T r2 = mx*mx + my*my;

    // calcul de mz
    T tmp = T(1.0) - (T(2.0)*alpha - T(1.0))*r2;
    if (tmp < eps) tmp = eps;

    T mz = (T(1.0) - alpha*alpha*r2) / (alpha * sqrt(tmp) + (T(1.0) - alpha) + eps);

    // dénominateur pour k
    T denom = mz*mz + r2;
    if (denom < eps) denom = eps;

    // k selon Usenko et al.
    T k = (mz*xi + sqrt(mz*mz + (T(1.0)-xi*xi)*r2 + eps)) / denom;

    ray[0] = k * mx;
    ray[1] = k * my;
    ray[2] = k * mz - xi;

    // normalisation
    T norm = sqrt(ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2] + eps);
    ray[0] /= norm;
    ray[1] /= norm;
    ray[2] /= norm;
}

#endif // CAMERA_MODEL_HPP