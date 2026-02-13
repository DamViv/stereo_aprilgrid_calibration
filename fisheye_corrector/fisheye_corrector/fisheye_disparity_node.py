import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from .dscamera import DSCamera
import yaml

class FisheyeDisparityNode(Node):
    def __init__(self):
        super().__init__('fisheye_disparity_node')

        # Paramètres
        self.declare_parameter('image_topic_left', '/cam_0/image_raw')
        self.declare_parameter('image_topic_right', '/cam_1/image_raw')
        self.declare_parameter('disparity_topic', '/stereo/disparity_image')
        self.declare_parameter('calib_file', 'stereo_calibration.yaml')
        self.declare_parameter('output_size', [512, 512])
        self.declare_parameter('method', 'perspective')

        image_topic_left = self.get_parameter('image_topic_left').get_parameter_value().string_value
        image_topic_right = self.get_parameter('image_topic_right').get_parameter_value().string_value
        disparity_topic = self.get_parameter('disparity_topic').get_parameter_value().string_value
        calib_file = self.get_parameter('calib_file').get_parameter_value().string_value
        self.output_size = tuple(self.get_parameter('output_size').get_parameter_value().integer_array_value)
        self.method = self.get_parameter('method').get_parameter_value().string_value        

        # load calib
        self.left_cam, self.right_cam, self.T_BS, self.rotvec, self.tvec = self.load_stereo_calib(calib_file)
        self.baseline = np.linalg.norm(self.tvec)

        # CvBridge
        self.bridge = CvBridge()

        # Charger les caméras
        self.left_cam = DSCamera(calib_file)
        self.right_cam = DSCamera(calib_file)

        # Souscriptions
        self.sub_left = self.create_subscription(Image, image_topic_left, self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, image_topic_right, self.right_callback, 10)

        # Stocker les dernières images
        self.left_img = None
        self.right_img = None

        # Créer l'algorithme stéréo
        blockSize = 7
        img_chanels = 3
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # multiple de 16
            blockSize=blockSize,
            P1=8*img_chanels*blockSize**2,  # small disparity change
            P2=32*img_chanels*blockSize**2, # huge disparity change
            disp12MaxDiff=1,                # L-R / R-L difference tolerance
            uniquenessRatio=10,
            speckleWindowSize=25,
            speckleRange=32
        )

    def load_stereo_calib(self, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        cam0_data = data['camera_0']
        cam1_data = data['camera_1']

        # Caméras
        left_cam = DSCamera(intrinsic=cam0_data['intrinsics'][0],
                            img_size=tuple(cam0_data['resolution'][0]))
        right_cam = DSCamera(intrinsic=cam1_data['intrinsics'][0],
                            img_size=tuple(cam1_data['resolution'][0]))

        # Transformation droite->gauche
        T_BS = np.array(data['T_BS']['data']).reshape((4, 4))

        # Rotation et translation (optionnel, tu peux reconstruire T_BS à partir de rotvec+tvec)
        rotvec = np.array(data['rotvec'])
        tvec = np.array(data['tvec'])

        return left_cam, right_cam, T_BS, rotvec, tvec


    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.compute_disparity_if_ready()

    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.compute_disparity_if_ready()


    def init_rectify_map_ds(self, K, alpha, xi, R, P, size):
        """Retourne les maps map1, map2 pour le modèle double-sphere"""
        w, h = size
        map1 = np.zeros((h, w), dtype=np.float32)
        map2 = np.zeros((h, w), dtype=np.float32)

        KRi = np.linalg.inv(P @ R)  # comme en C++

        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        s  = K[0,1] if K.shape[1] > 2 else 0.0

        for r in range(h):
            for c in range(w):
                # point dans le repère rectifié
                vec = np.array([c, r, 1.0])
                xc, yc, zc = (KRi @ vec).flatten()
                
                rr = np.sqrt(xc**2 + yc**2 + zc**2)
                xs, ys, zs = xc/rr, yc/rr, zc/rr
                
                d1 = np.sqrt(xs**2 + ys**2 + zs**2)
                d2 = np.sqrt(xs**2 + ys**2 + (xi*d1 + zs)**2)
                
                xd = xs / (alpha * d2 + (1-alpha)*(xi*d1 + zs))
                yd = ys / (alpha * d2 + (1-alpha)*(xi*d1 + zs))
                
                u = fx * xd + s * yd + cx
                v = fy * yd + cy
                
                map1[r, c] = u
                map2[r, c] = v

        return map1, map2










    def compute_disparity_if_ready(self):
        if self.left_img is None or self.right_img is None:
            return

        # Créer la caméra rectifiée virtuelle
        vfov = 90  # angle vertical souhaité en deg
        focal = self.output_size[1]/2 / np.tan(np.deg2rad(vfov)/2)
        Knew = np.array([[focal, 0, self.output_size[0]/2],
                        [0, focal, self.output_size[1]/2],
                        [0, 0, 1]])

        R_left = np.eye(3)  # gauche reste identité
        R_right = cv2.Rodrigues(-self.rotvec)[0]  # rotation droite -> gauche
        P_left = Knew.copy()
        P_right = Knew.copy()

        # Maps rectifiées exactes
        self.map_left, self.map_left2 = self.init_rectify_map_ds(Knew, 
                                                            self.left_cam.alpha, self.left_cam.xi, 
                                                            R_left, P_left, self.output_size)
        self.map_right, self.map_right2 = self.init_rectify_map_ds(Knew, 
                                                            self.right_cam.alpha, self.right_cam.xi, 
                                                            R_right, P_right, self.output_size)


        # Remap avec les maps exactes DS
        left_rect = cv2.remap(self.left_img, self.map_left, self.map_left2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(self.right_img, self.map_right, self.map_right2, cv2.INTER_LINEAR)

        disp16 = self.stereo.compute(cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)).astype(np.float32)/16.0

        fx = Knew[0,0]
        baseline = np.linalg.norm(self.tvec)
        depth = np.zeros_like(disp16, dtype=np.float32)
        valid = disp16 > 0

        depth[valid] = fx * baseline / disp16[valid]


        # left_rect et right_rect sont les images rectifiées
        cv2.imshow("Left Rectified", left_rect)
        cv2.imshow("Right Rectified", right_rect)
        cv2.imshow("disparity", disp16)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)



    def pc_from_depth(self, depth, K):
        h, w = depth.shape
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        pts = []
        for r in range(0, h, 4):  # sous-échantillonnage
            for c in range(0, w, 4):
                z = depth[r, c]
                if z <= 0:
                    continue
                x = (c - cx) / fx * z
                y = (r - cy) / fy * z
                pts.append([x, y, z])
        return np.array(pts)



def main(args=None):
    rclpy.init(args=args)
    node = FisheyeDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()