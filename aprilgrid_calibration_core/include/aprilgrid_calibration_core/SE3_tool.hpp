#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// Convert rotation matrix -> axis-angle (Lie log)
cv::Mat logSO3(const cv::Mat& R)
{
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    return rvec;
}

// Convert axis-angle -> rotation matrix (Lie exp)
cv::Mat expSO3(const cv::Mat& rvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return R;
}

// Compute component-wise median of translations
cv::Mat medianTranslation(const std::vector<cv::Mat>& Ts)
{
    std::vector<double> xs, ys, zs;
    for (const auto& t : Ts) {
        xs.push_back(t.at<double>(0));
        ys.push_back(t.at<double>(1));
        zs.push_back(t.at<double>(2));
    }

    auto median = [](std::vector<double>& v) {
        size_t n = v.size();
        std::nth_element(v.begin(), v.begin() + n / 2, v.end());
        return v[n / 2];
    };

    cv::Mat t_med = (cv::Mat_<double>(3,1) <<
        median(xs),
        median(ys),
        median(zs)
    );

    return t_med;
}

// MAIN FUNCTION
cv::Mat medianSE3(const std::vector<cv::Mat>& stereo_estimates)
{
    CV_Assert(!stereo_estimates.empty());

    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;

    // Extract R and t from each 4x4 SE3
    for (const auto& RT : stereo_estimates) {
        cv::Mat R = RT(cv::Rect(0,0,3,3)).clone();
        cv::Mat t = RT(cv::Rect(3,0,1,3)).clone();

        rotations.push_back(R);
        translations.push_back(t);
    }

    // Convert rotations to Lie algebra (rvec)
    std::vector<cv::Mat> rvecs;
    for (const auto& R : rotations)
        rvecs.push_back(logSO3(R));

    // Compute median rotation in Lie algebra
    cv::Mat rvec_med = cv::Mat::zeros(3,1,CV_64F);
    for (int k = 0; k < 3; k++) {
        std::vector<double> axis_values;
        for (auto& r : rvecs)
            axis_values.push_back(r.at<double>(k));

        std::nth_element(axis_values.begin(), axis_values.begin() + axis_values.size()/2, axis_values.end());
        rvec_med.at<double>(k) = axis_values[axis_values.size()/2];
    }

    // Back to SO3
    cv::Mat R_med = expSO3(rvec_med);

    // Median translation
    cv::Mat t_med = medianTranslation(translations);

    // Assemble SE3
    cv::Mat RT_med = cv::Mat::eye(4,4,CV_64F);
    R_med.copyTo(RT_med(cv::Rect(0,0,3,3)));
    t_med.copyTo(RT_med(cv::Rect(3,0,1,3)));

    return RT_med;
}