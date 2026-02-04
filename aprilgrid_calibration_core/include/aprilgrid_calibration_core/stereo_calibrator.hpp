#include "aprilgrid_calibration_core/acalibrator.hpp"

class StereoCalibrator : public ACalibrator
{
public:
    StereoCalibrator(int image_width,
                     int image_height,
                     CameraModel model = CameraModel::PINHOLE)
            : ACalibrator(image_width, image_height, model) {};
                

    void setIntrinsics(const cv::Mat& K0, const cv::Mat& D0,
                        const cv::Mat& K1, const cv::Mat& D1);

    void addView(const std::vector<cv::Point2f>& pts2D_cam0,
                 const std::vector<cv::Point2f>& pts2D_cam1,
                 const std::vector<cv::Point3f>& pts3D){
        image_0_points_.push_back(pts2D_cam0);
        image_1_points_.push_back(pts2D_cam1);
        object_points_.push_back(pts3D);
    }

    bool ready() const override {return !object_points_.empty();};
    bool calibrate() override;    

    void getExtrinsics(cv::Mat& R, cv::Mat &t) {    
        cv::Mat rvec = (cv::Mat_<double>(3,1) << RT_stereo_[0], RT_stereo_[1], RT_stereo_[2]);    
        cv::Rodrigues(rvec, R);    
        t = (cv::Mat_<double>(3,1) << RT_stereo_[3], RT_stereo_[4], RT_stereo_[5]);
    }
    
private:
    
    void initIntrinsics(const cv::Mat& K,  const cv::Mat& D, double intrinsics[6]);

    bool calibrateStereoDoubleSphere(
        double* intrinsec0,
        double* intrinsec1,
        std::vector<std::array<double, 6>>& RT_mire_views,
        std::array<double, 6>& RT_cam1_cam0,
        bool optim_intrinsecs);
    // bool calibrateStereoPinhole(double* intrinsics, std::vector<std::array<double,6>>& rts);
    // bool calibrateStereoFisheye(double* intrinsics, std::vector<std::array<double,6>>& rts);

    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<cv::Point2f>> image_0_points_;
    std::vector<std::vector<cv::Point2f>> image_1_points_;

    cv::Mat K0_, K1_;
    cv::Mat D0_, D1_;
    std::array<double, 6> RT_stereo_;
};
