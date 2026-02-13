import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json

from .dscamera import DSCamera

class FisheyeCorrectorNode(Node):
    def __init__(self):
        super().__init__('fisheye_corrector_node')

        # Paramètres ROS
        self.declare_parameter('image_topic', '/cam_0/image_raw')
        self.declare_parameter('corrected_topic', '/cam_0/image_corrected')
        self.declare_parameter('calib_file', 'calib.yaml')
        self.declare_parameter('output_size', [512, 512])
        self.declare_parameter('method', 'perspective')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        corrected_topic = self.get_parameter('corrected_topic').get_parameter_value().string_value
        calib_file = self.get_parameter('calib_file').get_parameter_value().string_value
        self.output_size = tuple(self.get_parameter('output_size').get_parameter_value().integer_array_value)
        self.method = self.get_parameter('method').get_parameter_value().string_value

        # Initialisation DSCamera
        self.camera = DSCamera(yaml_filename=calib_file)

        # CvBridge
        self.bridge = CvBridge()

        # Subscriber
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Publisher
        self.pub = self.create_publisher(
            Image,
            corrected_topic,
            10
        )

        self.get_logger().info(f"Node initialized. Subscribing to {image_topic}, publishing corrected images on {corrected_topic}")

    def image_callback(self, msg: Image):
        # Convertir ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Corriger l'image
        if self.method =='perspective':
            corrected_img = self.camera.to_perspective(cv_image, img_size=self.output_size)
        if self.method =='equirect':
            corrected_img = self.camera.to_equirect(cv_image, img_size=self.output_size)

        # Convertir OpenCV -> ROS Image
        try:
            corrected_msg = self.bridge.cv2_to_imgmsg(corrected_img.astype(np.uint8), encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting corrected image: {e}")
            return

        # Publier l'image corrigée
        self.pub.publish(corrected_msg)
        self.get_logger().debug("Published corrected image")

def main(args=None):
    rclpy.init(args=args)
    node = FisheyeCorrectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()