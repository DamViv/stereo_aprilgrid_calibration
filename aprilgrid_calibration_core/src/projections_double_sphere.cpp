#include "../include/aprilgrid_calibration_core/projections_double_sphere.hpp"

// Projection equirectangular
cv::Mat generateEquirectangular(const cv::Mat& img, const cv::Mat& K, const cv::Mat& D, int width, int height) {
    cv::Mat map_x(height, width, CV_32F);
    cv::Mat map_y(height, width, CV_32F);

    double* intrinsics = new double[6];
    intrinsics[0] = K.at<double>(0,0); // fx
    intrinsics[1] = K.at<double>(1,1); // fy
    intrinsics[2] = K.at<double>(0,2); // cx
    intrinsics[3] = K.at<double>(1,2); // cy
    intrinsics[4] = D.at<double>(0);   // xi
    intrinsics[5] = D.at<double>(1);   // alpha

    for(int j=0; j<height; j++){
        double phi = 2*M_PI/3 * (0.5 - static_cast<double>(j)/height); // latitude [-pi/2, pi/2]

        for(int i=0; i<width; i++){
            double theta = M_PI * (static_cast<double>(i)/width - 0.5); // longitude [-pi, pi]

            // Coordonnées unitaires en caméra
            double x = cos(phi) * sin(theta);
            double y = sin(phi);
            double z = cos(phi) * cos(theta);
            cv::Vec3d P(x,y,z);

            // Projection DS
            double u, v;
            projectDoubleSphere(P[0], P[1], P[2], intrinsics, u, v);            

            map_x.at<float>(j,i) = u;
            map_y.at<float>(j,i) = v;
        }
    }

    cv::Mat equi;
    cv::remap(img, equi, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return equi;
}


cv::Mat undistortDoubleSphere(const cv::Mat& img, const cv::Mat& K, const cv::Mat& D) {
    cv::Mat map_x(img.rows, img.cols, CV_32F);
    cv::Mat map_y(img.rows, img.cols, CV_32F);

    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);
    double xi = D.at<double>(0);
    double alpha = D.at<double>(1);
    
    // Nouvelle matrice (ajuster scale si besoin)
    double fx_new = fx * 0.6;  // Réduire pour voir plus
    double fy_new = fy * 0.6;
    double cx_new = img.cols / 2.0;
    double cy_new = img.rows / 2.0;

    for(double v = 0; v < img.rows; v++) {
        for(double u = 0; u < img.cols; u++) {
            // Unprojection perspective
            double x = (u - cx_new) / fx_new;
            double y = (v - cy_new) / fy_new;
            double z = 1.0;
            
            // Normaliser
            double norm = sqrt(x*x + y*y + z*z);
            x /= norm; y /= norm; z /= norm;
            
            // Reprojection Double Sphere
            projectDoubleSphere(x, y, z, new double[6]{fx, fy, cx, cy, xi, alpha}, u, v);
                            
            map_x.at<float>(v, u) = u;
            map_y.at<float>(v, u) = v;            
        }
    }

    cv::Mat undistorted;
    cv::remap(img, undistorted, map_x, map_y, cv::INTER_LINEAR);
    return undistorted;
}
