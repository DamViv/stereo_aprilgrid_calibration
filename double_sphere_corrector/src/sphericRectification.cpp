#include "../include/sphericRectification.hpp"
#include <yaml-cpp/yaml.h>


sphericRectification::sphericRectification(std::string configfilepath){
    loadStereoConfig(configfilepath);
}


void sphericRectification::loadStereoConfig(const std::string& config_file)
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
    YAML::Node intr0 = cam0["intrinsics"][0];
    YAML::Node res0 = cam0["resolution"][0];
    double fxl = intr0["fx"].as<double>();
    double fyl = intr0["fy"].as<double>();
    double cxl = intr0["cx"].as<double>();
    double cyl = intr0["cy"].as<double>();
    xil_ = intr0["xi"].as<double>();
    alphal_ = intr0["alpha"].as<double>();

    Kl_ = (cv::Mat_<double>(3,3) <<
                  fxl, 0,   cxl,
                  0,   fyl, cyl,
                  0,   0,   1);

    img_size_ =cv::Size(res0[0].as<int>(), res0[1].as<int>());

    // ---------------- Cam1 ----------------
    YAML::Node cam1 = config["camera_1"];
    YAML::Node intr1 = cam1["intrinsics"][0];

    double fxr = intr1["fx"].as<double>();
    double fyr = intr1["fy"].as<double>();
    double cxr = intr1["cx"].as<double>();
    double cyr = intr1["cy"].as<double>();
    xir_ = intr1["xi"].as<double>();
    alphar_ = intr1["alpha"].as<double>();

    Kr_ = (cv::Mat_<double>(3,3) <<
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

    Rstereo_ = T_BS(cv::Rect(0,0,3,3)).clone();    
    Tstereo_ = T_BS(cv::Rect(3,0,1,3)).clone();
}



void sphericRectification::processImages(const cv::Mat& Il, const cv::Mat& Ir){

    // --- 1. Dimensions et FOV ---
    int width  = 1024;  // largeur image rectifiée
    int height = 768;   // hauteur image rectifiée
    double FOVx_deg = 180.0;
    double FOVy_deg = 180.0;

    // --- 2. Maps de remap ---
    cv::Mat map_x_l, map_y_l, map_x_r, map_y_r;
    computeSphericalMaps(width, height, map_x_l, map_y_l, map_x_r, map_y_r, FOVx_deg, FOVy_deg);

    // --- 3. Rectification des images ---
    cv::Mat Il_rect, Ir_rect;
    cv::remap(Il, Il_rect, map_x_l, map_y_l, cv::INTER_LINEAR);
    cv::remap(Ir, Ir_rect, map_x_r, map_y_r, cv::INTER_LINEAR);

    cv::imshow("Il_rect", Il_rect);
    cv::imshow("Ir_rect", Ir_rect);
    cv::waitKey(1);

    // --- 4. Préparer nuage de points ---
    std::vector<cv::Point3d> cloud3D;
    cloud3D.reserve(width*height);

    // // --- 5. Stéréo très basique : assume correspondance simple horizontale ---
    // // (ici juste pour l'exemple, tu peux remplacer par SGBM ou feature matching)
    // for(int v=0; v<height; v++){
    //     for(int u=0; u<width; u++){
    //         // Rayon gauche
    //         cv::Vec3d r_left = unprojectDS(cv::Vec2d(u,v), Kl_, xil_, alphal_);

    //         // Rayon droit : correspondance horizontale basique
    //         int u_right = u; // placeholder, remplacer par vraie disparité
    //         int v_right = v;
    //         cv::Vec3d r_right = unprojectDS(cv::Vec2d(u_right,v_right), Kr_, xir_, alphar_);

    //         // Triangulation
    //         cv::Point3d P = triangulateRay(r_left, r_right, cv::Vec3d(0,0,0), Tstereo_);
    //         cloud3D.push_back(P);
    //     }
    // }

    // std::cout << "Nuage 3D généré : " << cloud3D.size() << " points." << std::endl;


}



void sphericRectification::computeSphericalMaps(int width, int height, 
                                                cv::Mat& map_x_l, cv::Mat& map_y_l,
                                                cv::Mat& map_x_r, cv::Mat& map_y_r,
                                                double FOVx_deg, double FOVy_deg)
{
    // --- Convertir FOV en radians ---
    double FOVx = FOVx_deg * CV_PI / 180.0;
    double FOVy = FOVy_deg * CV_PI / 180.0;

    // --- Initialiser les maps ---
    map_x_l = cv::Mat(height, width, CV_32F);
    map_y_l = cv::Mat(height, width, CV_32F);
    map_x_r = cv::Mat(height, width, CV_32F);
    map_y_r = cv::Mat(height, width, CV_32F);

    // --- Pour chaque pixel de l'image rectifiée sphérique ---
    for(int v=0; v<height; v++){
        for(int u=0; u<width; u++){

            // --- 1. Calcul des angles sphériques ---
            // u,v -> [-FOVx/2,FOVx/2], [-FOVy/2,FOVy/2]
            double theta = ((double)u / (width-1) - 0.5) * FOVx;   // angle horizontal
            double phi   = ((double)v / (height-1) - 0.5) * FOVy;  // angle vertical

            // --- 2. Générer le rayon 3D unitaire correspondant ---
            cv::Vec3d ray;
            ray[0] = cos(phi) * sin(theta);  // x
            ray[1] = sin(phi);               // y
            ray[2] = cos(phi) * cos(theta);  // z

            // --- 3. Projection forward dans la caméra gauche ---
            cv::Vec2d uv_l = projectDS(ray, Kl_, xil_, alphal_);
            map_x_l.at<float>(v,u) = static_cast<float>(uv_l[0]);
            map_y_l.at<float>(v,u) = static_cast<float>(uv_l[1]);

            // --- 4. Transformation du rayon dans le repère de la caméra droite ---
            //cv::Vec3d ray_r = Rstereo_ * ray + Tstereo_;
            cv::Vec3d ray_r;
            ray_r[0] = Rstereo_.at<double>(0,0)*ray[0] + Rstereo_.at<double>(0,1)*ray[1] + Rstereo_.at<double>(0,2)*ray[2] + Tstereo_.at<double>(0);
            ray_r[1] = Rstereo_.at<double>(1,0)*ray[0] + Rstereo_.at<double>(1,1)*ray[1] + Rstereo_.at<double>(1,2)*ray[2] + Tstereo_.at<double>(1);
            ray_r[2] = Rstereo_.at<double>(2,0)*ray[0] + Rstereo_.at<double>(2,1)*ray[1] + Rstereo_.at<double>(2,2)*ray[2] + Tstereo_.at<double>(2);


            // --- 5. Projection forward dans la caméra droite ---
            cv::Vec2d uv_r = projectDS(ray_r, Kr_, xir_, alphar_);
            map_x_r.at<float>(v,u) = static_cast<float>(uv_r[0]);
            map_y_r.at<float>(v,u) = static_cast<float>(uv_r[1]);
        }
    }
}




cv::Vec3d sphericRectification::unprojectDS(const cv::Vec2d& uv, const cv::Mat& K, double xi, double alpha)
{
    const double fx = K.at<double>(0,0);
    const double fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2);
    const double cy = K.at<double>(1,2);

    const double eps = double(1e-8);

    // coordonnées normalisées
    double mx = (uv[0] - cx) / fx;
    double my = (uv[1] - cy) / fy;
    double r2 = mx*mx + my*my;

    // calcul de mz
    double tmp = 1.0 - (2.0*alpha - 1.0)*r2;
    if (tmp < eps) tmp = eps;

    double mz = (1.0 - alpha*alpha*r2) / (alpha * sqrt(tmp) + (1.0 - alpha) + eps);

    // dénominateur pour k
    double denom = mz*mz + r2;
    if (denom < eps) denom = eps;

    // k selon Usenko et al.
    double k = (mz*xi + sqrt(mz*mz + (1.0-xi*xi)*r2 + eps)) / denom;

    cv::Vec3d ray;
    ray[0] = k * mx;
    ray[1] = k * my;
    ray[2] = k * mz - xi;

    // normalisation
    double norm = sqrt(ray[0]*ray[0] + ray[1]*ray[1] + ray[2]*ray[2] + eps);
    ray[0] /= norm;
    ray[1] /= norm;
    ray[2] /= norm;

    return ray;
}

cv::Vec2d sphericRectification::projectDS(const cv::Vec3d& ray, const cv::Mat& K, double xi, double alpha)
{
    // Rayon 3D non-normalisé
    double X = ray[0];
    double Y = ray[1];
    double Z = ray[2];

    const double fx = K.at<double>(0,0);
    const double fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2);
    const double cy = K.at<double>(1,2);

    const double eps = 1e-8;

    // norme du vecteur projeté sur le plan xy
    double d = sqrt(X*X + Y*Y + eps);

    // mz selon Usenko et al. (pour projection double-sphere)
    double mz = (Z + xi * sqrt(X*X + Y*Y + Z*Z)) / (1.0 - alpha);

    // Projection finale
    double mx = X / (mz + eps);
    double my = Y / (mz + eps);

    // pixel coordinates
    double u = fx * mx + cx;
    double v = fy * my + cy;

    return cv::Vec2d(u, v);
}