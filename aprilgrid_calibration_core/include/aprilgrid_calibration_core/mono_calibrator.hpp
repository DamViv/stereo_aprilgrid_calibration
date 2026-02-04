#include "aprilgrid_calibration_core/acalibrator.hpp"

class MonoCalibrator : public ACalibrator
{
public:
    MonoCalibrator(int image_width,
                   int image_height,
                   CameraModel model = CameraModel::PINHOLE)
            : ACalibrator(image_width, image_height, model) {};

    void addView(const std::vector<cv::Point2f>& pts2D,
                 const std::vector<cv::Point3f>& pts3D);

    bool ready() const override;
    bool calibrate() override;

    const cv::Mat& cameraMatrix() const;
    const cv::Mat& distCoeffs() const;
    

private:

    bool calibrate_double_sphere(double* intrinsics, std::vector<std::array<double,6>>& rts);
    bool calibrate_pinhole(double* intrinsics, std::vector<std::array<double,6>>& rts);
    bool calibrate_fisheye(double* intrinsics, std::vector<std::array<double,6>>& rts);

    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<cv::Point2f>> image_points_;

    cv::Mat K_;
    cv::Mat D_;
};
