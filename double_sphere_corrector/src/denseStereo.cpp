#include "../include/denseStereo.hpp"
#include <opencv2/video/tracking.hpp>
#include <yaml-cpp/yaml.h>


inline double MatRowMul(cv::Mat m, double x, double y, double z, int r) {
    return m.at<double>(r, 0) * x + m.at<double>(r, 1) * y + m.at<double>(r, 2) * z;
}

denseStereo::denseStereo(std::string configfilepath) : _configfilepath(configfilepath) {

    // Load parameters
    loadStereoConfig(configfilepath);
}

void denseStereo::loadStereoConfig(const std::string& config_file)
{
    YAML::Node config;

    try {
        config = YAML::LoadFile(config_file);
    } catch (const std::exception &e) {
        std::cerr << "Failed to open YAML file: " << e.what() << std::endl;
        exit(-1);
    }

    // ---------------- Cam0 ----------------
    YAML::Node cam0 = config["camera_0"];
    _cam_model = cam0["camera_type"].as<std::string>();

    YAML::Node intr0 = cam0["intrinsics"][0];
    double fxl = intr0["fx"].as<double>();
    double fyl = intr0["fy"].as<double>();
    double cxl = intr0["cx"].as<double>();
    double cyl = intr0["cy"].as<double>();
    xil = intr0["xi"].as<double>();
    alphal = intr0["alpha"].as<double>();

    YAML::Node res0 = cam0["resolution"][0];
    int width = res0[0].as<int>();
    int height = res0[1].as<int>();
    
    _cap_cols = width;
    _cap_rows = height;
    _width    = _cap_cols;
    _height   = _cap_rows;

    Kl = (cv::Mat_<double>(3,3) <<
                  fxl, 0,   cxl,
                  0,   fyl, cyl,
                  0,   0,   1);

    Rl = cv::Mat::eye(3, 3, CV_64F);

    // ---------------- Cam1 ----------------
    YAML::Node cam1 = config["camera_1"];
    YAML::Node intr1 = cam1["intrinsics"][0];

    double fxr = intr1["fx"].as<double>();
    double fyr = intr1["fy"].as<double>();
    double cxr = intr1["cx"].as<double>();
    double cyr = intr1["cy"].as<double>();
    xir = intr1["xi"].as<double>();
    alphar = intr1["alpha"].as<double>();

    Kr = (cv::Mat_<double>(3,3) <<
                  fxr, 0,   cxr,
                  0,   fyr, cyr,
                  0,   0,   1);

    // ---------------- T_BS ----------------
    YAML::Node Tnode = config["T_BS"]["data"];
    if(Tnode.size() != 16){
        std::cerr << "T_BS data invalid!" << std::endl;
        exit(-1);
    }
    cv::Mat T_BS(4,4,CV_64F);
    for(int i=0;i<16;i++){
        T_BS.at<double>(i/4, i%4) = Tnode[i].as<double>();
    }

    Rr = T_BS(cv::Rect(0,0,3,3)).clone();    
    Translation = T_BS(cv::Rect(3,0,1,3)).clone();
}


void denseStereo::InitUndistortRectifyMap(cv::Mat K,
                                          cv::Mat D,
                                          double xi,
                                          cv::Mat R,
                                          cv::Mat P,
                                          cv::Size size,
                                          cv::Mat &map1,
                                          cv::Mat &map2) {
    map1 = cv::Mat(size, CV_32F);
    map2 = cv::Mat(size, CV_32F);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double s  = K.at<double>(0, 1);

    double xid = xi;

    double k1 = D.at<double>(0, 0);
    double k2 = D.at<double>(0, 1);
    double p1 = D.at<double>(0, 2);
    double p2 = D.at<double>(0, 3);

    cv::Mat KRi = (P * R).inv();

    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            double xc = MatRowMul(KRi, c, r, 1., 0);
            double yc = MatRowMul(KRi, c, r, 1., 1);
            double zc = MatRowMul(KRi, c, r, 1., 2);

            double rr = sqrt(xc * xc + yc * yc + zc * zc);
            double xs = xc / rr;
            double ys = yc / rr;
            double zs = zc / rr;

            double xu = xs / (zs + xid);
            double yu = ys / (zs + xid);

            double r2 = xu * xu + yu * yu;
            double r4 = r2 * r2;
            double xd = (1 + k1 * r2 + k2 * r4) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu);
            double yd = (1 + k1 * r2 + k2 * r4) * yu + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu * yu);

            double u = fx * xd + s * yd + cx;
            double v = fy * yd + cy;

            map1.at<float>(r, c) = (float)u;
            map2.at<float>(r, c) = (float)v;
        }
    }
}

void denseStereo::InitUndistortRectifyMapDS(cv::Mat K,
                                            float alpha,
                                            float xi,
                                            cv::Mat R,
                                            cv::Mat P,
                                            cv::Size size,
                                            cv::Mat &map1,
                                            cv::Mat &map2) {
    map1 = cv::Mat(size, CV_32F);
    map2 = cv::Mat(size, CV_32F);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double s  = K.at<double>(0, 1);

    cv::Mat KRi = (P * R).inv();

    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            double xc = MatRowMul(KRi, c, r, 1., 0);
            double yc = MatRowMul(KRi, c, r, 1., 1);
            double zc = MatRowMul(KRi, c, r, 1., 2);

            double rr = sqrt(xc * xc + yc * yc + zc * zc);
            double xs = xc / rr;
            double ys = yc / rr;
            double zs = zc / rr;

            double d1 = std::sqrt(xs * xs + ys * ys + zs * zs);
            double d2 = std::sqrt(xs * xs + ys * ys + (xi * d1 + zs) * (xi * d1 + zs));

            double xd = xs / (alpha * d2 + (1 - alpha) * (xi * d1 + zs));
            double yd = ys / (alpha * d2 + (1 - alpha) * (xi * d1 + zs));

            double u = fx * xd + s * yd + cx;
            double v = fy * yd + cy;

            map1.at<float>(r, c) = (float)u;
            map2.at<float>(r, c) = (float)v;
        }
    }
}

void denseStereo::InitRectifyMap() {

    double vfov_rad = _vfov * CV_PI / 180.;
    double focal    = _height / 2. / tan(vfov_rad / 2.);
    Knew = (cv::Mat_<double>(3, 3) << focal, 0., _width / 2. - 0.5, 0., focal, _height / 2. - 0.5, 0., 0., 1.);

    cv::Size img_size(_width, _height);

    if (_cam_model == "ds") {
        InitUndistortRectifyMapDS(
            Kl, alphal, xil, Rl, Knew, img_size, smap[0][0], smap[0][1]);
        InitUndistortRectifyMapDS(
            Kr, alphar, xir, Rr, Knew, img_size, smap[1][0], smap[1][1]);
    } else if (_cam_model == "omni") {
        InitUndistortRectifyMap(Kl, Dl, xil, Rl, Knew, img_size, smap[0][0], smap[0][1]);
        InitUndistortRectifyMap(Kr, Dr, xir, Rr, Knew, img_size, smap[1][0], smap[1][1]);
    }    

}

pcl::PointCloud<pcl::PointXYZ>::Ptr denseStereo::Triangulate(const cv::Mat &recl, const cv::Mat &recr) {

    std::vector<cv::Point2f> ptsLeft, ptsRight;
    cv::goodFeaturesToTrack(recl, ptsLeft, 1000, 0.01, 10);
    cv::goodFeaturesToTrack(recr, ptsRight, 1000, 0.01, 10);

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(recl, recr, ptsLeft, ptsRight, status, err);

    cv::Mat R = Rr.t(); // rotation relative droite ← gauche
    cv::Mat T = -Translation; // translation droite ← gauche


    // Cam gauche : P1 = Kl * [I | 0]
    cv::Mat P1 = cv::Mat::zeros(3,4,CV_64F);
    Kl.copyTo(P1(cv::Rect(0,0,3,3))); // met Kl dans la partie gauche
    // La dernière colonne reste 0 pour translation

    // Cam droite : P2 = Kr * [R | T]
    cv::Mat P2 = cv::Mat::zeros(3,4,CV_64F);
    R.copyTo(P2(cv::Rect(0,0,3,3))); // rotation
    T.copyTo(P2(cv::Rect(3,0,1,3))); // translation
    P2 = Kr * P2;

    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, ptsLeft, ptsRight, pts4D);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);

    int numPoints = pts4D.cols;
    pointcloud->width = numPoints;
    pointcloud->height = 1; // cloud non organisé
    pointcloud->is_dense = false;
    pointcloud->points.resize(numPoints);

    for(int i = 0; i < numPoints; ++i) {
        float w = pts4D.at<float>(3, i);
        if (w != 0.0f) {
            float x = pts4D.at<float>(0, i) / w;
            float y = pts4D.at<float>(1, i) / w;
            float z = pts4D.at<float>(2, i) / w;
            pointcloud->points[i] = pcl::PointXYZ(x, y, z);
        } else {
            // si W=0, met un point invalide
            pointcloud->points[i] = pcl::PointXYZ(std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::quiet_NaN(),
                                                std::numeric_limits<float>::quiet_NaN());
        }
    }

    return pointcloud;
}


void denseStereo::DisparityImage(const cv::Mat &recl, const cv::Mat &recr, cv::Mat &disp, cv::Mat &depth_map) {
    cv::Mat disp16s;
    int N = _ndisp, W = _wsize, C = recl.channels();
    if (is_sgbm) {
        cv::Ptr<cv::StereoSGBM> sgbm =
            cv::StereoSGBM::create(2, N, W, 8 * C * W * W, 32 * C * W * W, 0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM);
        sgbm->compute(recl, recr, disp16s);
    } else {

        cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(N, W);
        sbm->setPreFilterCap(31);
        sbm->setMinDisparity(0);
        sbm->setTextureThreshold(10);
        sbm->setUniquenessRatio(15);
        sbm->setSpeckleWindowSize(100);
        sbm->setSpeckleRange(32);
        sbm->setDisp12MaxDiff(1);
        sbm->compute(recl, recr, disp16s);
        
    }

    double minVal, maxVal;
    minMaxLoc(disp16s, &minVal, &maxVal);
    disp16s.convertTo(disp, CV_8UC1, 255 / (maxVal - minVal));

    // How to get the depth map
    double fx = Knew.at<double>(0, 0);
    double bl = cv::norm(Translation);

    cv::Mat dispf;
    disp16s.convertTo(dispf, CV_32F, 1.0/16.0f);
    depth_map = cv::Mat(dispf.rows, dispf.cols, CV_32F);

    for (int r = 0; r < dispf.rows; ++r) {
        for (int c = 0; c < dispf.cols; ++c) {

            double disp = dispf.at<float>(r, c);
            if (disp <= 0.f) {
                depth_map.at<float>(r, c) = 0.f;
            } else {
                double depth              = fx * bl / disp;
                depth_map.at<float>(r, c) = static_cast<float>(depth);
            }
        }
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr denseStereo::pcFromDepthMap(const cv::Mat &depth_map) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    double f_x = Knew.at<double>(0, 0);
    double f_y = Knew.at<double>(1, 1);
    double c_x = Knew.at<double>(0, 2);
    double c_y = Knew.at<double>(1, 2);

    for (int r = 0; r < depth_map.rows; ++r) {
    	if (r%4 != 0)
    	    continue;
        for (int c = 0; c < depth_map.cols; ++c) {
            if (c%4 != 0)
    	        continue;
            
            // The depth is the z value of the 3D point in the camera frame
            double w = static_cast<double>(depth_map.at<float>(r, c));
                
            // The 3D point is X = w  K^{-1} [u, v, 1]^T 
            double x = (double)(c - c_x) / f_x;
            double y = (double)(r - c_y) / f_y;
            double z = (double)1 ;

            pcl::PointXYZ pt;
            pt.x = x * w;
            pt.y = y * w;
            pt.z = z * w;

            pointcloud->points.push_back(pt);
        }
    }

    return pointcloud;
}
