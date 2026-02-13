#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QGroupBox, QProgressBar
)

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage

import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from aprilgrid_detector_interfaces.msg import CalibrationStats, AprilTagArray

from aprilgrid_detector_interfaces.srv import (
    StartCollection, StopCollection, ClearFrames,
    CalibrateCamera, CalibrateStereo
)


# Mapping pour StartCollection
COLLECTION_MODES = {
    "idle": 0,
    "mono_cam0": 1,
    "mono_cam1": 2,
    "stereo": 3,    
}


class CalibrationGUINode(Node):
    def __init__(self):
        super().__init__("aprilgrid_calibration_gui")

        self.app = QApplication(sys.argv)
        self.window = CalibrationWindow(self)

        self.window.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.spin_ros)
        self.timer.start(10)

    def spin_ros(self):
        rclpy.spin_once(self, timeout_sec=0.0)


class CalibrationWindow(QMainWindow):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node

        self.setWindowTitle("AprilGrid Calibration GUI")
        self.resize(1400, 800)

        self.bridge = CvBridge()

        self.last_img0 = None
        self.last_img1 = None
        self.tags0 = []
        self.tags1 = []

        self.cam0_frames = 0
        self.cam1_frames = 0
        self.stereo_pairs = 0

        self.build_ui()
        self.setup_ros()
        self.setup_timer()

    # ---------- UI ----------
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout()

        # ---------------- Cam0 Tabs ----------------
        self.cam0_tab = QTabWidget()
        self.cam0_raw_label = QLabel("Raw")
        self.cam0_anno_label = QLabel("Detection")
        self.cam0_corr_label = QLabel("Corrected")
        for lbl in [self.cam0_raw_label, self.cam0_anno_label, self.cam0_corr_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 240)
        self.cam0_tab.addTab(self.cam0_raw_label, "Raw")
        self.cam0_tab.addTab(self.cam0_anno_label, "Detection")
        self.cam0_tab.addTab(self.cam0_corr_label, "Corrected")
        layout.addWidget(self.cam0_tab, 0, 0)

        # ---------------- Cam1 Tabs ----------------
        self.cam1_tab = QTabWidget()
        self.cam1_raw_label = QLabel("Raw")
        self.cam1_anno_label = QLabel("Detection")
        self.cam1_corr_label = QLabel("Corrected")
        for lbl in [self.cam1_raw_label, self.cam1_anno_label, self.cam1_corr_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 240)
        self.cam1_tab.addTab(self.cam1_raw_label, "Raw")
        self.cam1_tab.addTab(self.cam1_anno_label, "Detection")
        self.cam1_tab.addTab(self.cam1_corr_label, "Corrected")
        layout.addWidget(self.cam1_tab, 0, 1)

        # ---------------- Stats ----------------
        layout = self.build_stats_ui(layout)

        # ---------------- Controls ----------------
        control_box = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        self.collecting_mono0 = False
        self.collecting_mono1 = False
        self.collecting_stereo = False

        self.btn_start_mono0 = QPushButton("Start Mono Cam0")
        self.btn_start_mono1 = QPushButton("Start Mono Cam1")
        self.btn_start_stereo = QPushButton("Start Stereo")
        self.btn_clear = QPushButton("Clear Frames")

        for btn in [self.btn_start_mono0, self.btn_start_mono1, self.btn_start_stereo, self.btn_clear]:
            control_layout.addWidget(btn)

        control_box.setLayout(control_layout)
        layout.addWidget(control_box, 2, 0, 1, 2)

        # ---------------- Calibration ----------------
        calib_box = QGroupBox("Calibration")
        calib_layout = QHBoxLayout()

        # --- Cam0 calib ---
        cam0_layout = QVBoxLayout()
        self.btn_calib_cam0 = QPushButton("Calibrate Cam0")
        self.calib_results_cam0_text = QLabel("No calibration yet")
        self.calib_results_cam0_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.calib_results_cam0_text.setMinimumWidth(200)
        cam0_layout.addWidget(self.btn_calib_cam0)
        cam0_layout.addWidget(self.calib_results_cam0_text)
        calib_layout.addLayout(cam0_layout)

        # --- Cam1 calib ---
        cam1_layout = QVBoxLayout()
        self.btn_calib_cam1 = QPushButton("Calibrate Cam1")
        self.calib_results_cam1_text = QLabel("No calibration yet")
        self.calib_results_cam1_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.calib_results_cam1_text.setMinimumWidth(200)
        cam1_layout.addWidget(self.btn_calib_cam1)
        cam1_layout.addWidget(self.calib_results_cam1_text)
        calib_layout.addLayout(cam1_layout)

        # --- Stereo calib ---
        stereo_layout = QVBoxLayout()
        self.btn_calib_stereo = QPushButton("Calibrate Stereo")
        self.calib_results_stereo_text = QLabel("No calibration yet")
        self.calib_results_stereo_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.calib_results_stereo_text.setMinimumWidth(200)
        stereo_layout.addWidget(self.btn_calib_stereo)
        stereo_layout.addWidget(self.calib_results_stereo_text)
        calib_layout.addLayout(stereo_layout)

        calib_box.setLayout(calib_layout)
        layout.addWidget(calib_box, 3, 0, 1, 2)

        # ---------------- Log ----------------
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 4, 0, 1, 2)

        # ---------------- Connections ----------------
        self.btn_start_mono0.clicked.connect(self.toggle_mono0_collection)
        self.btn_start_mono1.clicked.connect(self.toggle_mono1_collection)
        self.btn_start_stereo.clicked.connect(self.toggle_stereo_collection)
        self.btn_clear.clicked.connect(self.clear_frames)

        self.btn_calib_cam0.clicked.connect(lambda: self.calibrate_camera(0))
        self.btn_calib_cam1.clicked.connect(lambda: self.calibrate_camera(1))
        self.btn_calib_stereo.clicked.connect(self.calibrate_stereo)

        central.setLayout(layout)


    def build_stats_ui(self, parent_layout):
        stats_box = QGroupBox("Calibration Stats")
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(25)

        def make_bar():
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFixedHeight(14)
            bar.setMaximumWidth(220)
            bar.setTextVisible(False)
            return bar

        # --- CAM0 ---
        self.cam0_box = QGroupBox("Cam 0")
        cam0_layout = QVBoxLayout()

        self.cam0_frames_bar = make_bar()
        self.cam0_frames_label = QLabel("0 valid frames")
        self.cam0_frames_label.setAlignment(Qt.AlignCenter)

        self.cam0_div_bar = make_bar()
        self.cam0_cov_bar = make_bar()

        # Frames
        cam0_layout.addWidget(self.cam0_frames_bar, alignment=Qt.AlignCenter)
        cam0_layout.addWidget(self.cam0_frames_label)

        # Diversity
        div_label = QLabel("Diversity")
        div_label.setAlignment(Qt.AlignCenter)
        cam0_layout.addWidget(div_label)
        cam0_layout.addWidget(self.cam0_div_bar, alignment=Qt.AlignCenter)

        # Coverage
        cov_label = QLabel("Coverage")
        cov_label.setAlignment(Qt.AlignCenter)
        cam0_layout.addWidget(cov_label)
        cam0_layout.addWidget(self.cam0_cov_bar, alignment=Qt.AlignCenter)

        self.cam0_box.setLayout(cam0_layout)

        # --- CAM1 ---
        self.cam1_box = QGroupBox("Cam 1")
        cam1_layout = QVBoxLayout()

        self.cam1_frames_bar = make_bar()
        self.cam1_frames_label = QLabel("0 valid frames")
        self.cam1_frames_label.setAlignment(Qt.AlignCenter)

        self.cam1_div_bar = make_bar()
        self.cam1_cov_bar = make_bar()

        cam1_layout.addWidget(self.cam1_frames_bar, alignment=Qt.AlignCenter)
        cam1_layout.addWidget(self.cam1_frames_label)

        div_label = QLabel("Diversity")
        div_label.setAlignment(Qt.AlignCenter)
        cam1_layout.addWidget(div_label)
        cam1_layout.addWidget(self.cam1_div_bar, alignment=Qt.AlignCenter)

        cov_label = QLabel("Coverage")
        cov_label.setAlignment(Qt.AlignCenter)
        cam1_layout.addWidget(cov_label)
        cam1_layout.addWidget(self.cam1_cov_bar, alignment=Qt.AlignCenter)

        self.cam1_box.setLayout(cam1_layout)

        # --- STEREO ---
        self.stereo_box = QGroupBox("Stereo")
        stereo_layout = QVBoxLayout()

        self.stereo_frames_bar = make_bar()
        self.stereo_frames_label = QLabel("0 valid stereo pairs")
        self.stereo_frames_label.setAlignment(Qt.AlignCenter)

        stereo_layout.addWidget(self.stereo_frames_bar, alignment=Qt.AlignCenter)
        stereo_layout.addWidget(self.stereo_frames_label)

        self.stereo_box.setLayout(stereo_layout)

        # --- Add to main stats layout ---
        stats_layout.addWidget(self.cam0_box)
        stats_layout.addWidget(self.cam1_box)
        stats_layout.addWidget(self.stereo_box)

        stats_box.setLayout(stats_layout)
        parent_layout.addWidget(stats_box, 1, 0, 1, 2)

        return parent_layout


    # ---------- ROS ----------
    def setup_ros(self):
        n = self.node

        n.create_subscription(Image, "/cam_0/image_raw", self.image0_cb, qos_profile_sensor_data)
        n.create_subscription(Image, "/cam_1/image_raw", self.image1_cb, qos_profile_sensor_data)

        n.create_subscription(Image, "/cam_0/apriltags_img", self.anno0_cb, qos_profile_sensor_data)
        n.create_subscription(Image, "/cam_1/apriltags_img", self.anno1_cb, qos_profile_sensor_data)

        n.create_subscription(Image, "/cam_0/image_corrected", self.image0_corr_cb, qos_profile_sensor_data)
        n.create_subscription(Image, "/cam_1/image_corrected", self.image1_corr_cb, qos_profile_sensor_data)

        n.create_subscription(AprilTagArray, "/cam_0/apriltags", self.tags0_cb, qos_profile_sensor_data)
        n.create_subscription(AprilTagArray, "/cam_1/apriltags", self.tags1_cb, qos_profile_sensor_data)

        n.create_subscription(CalibrationStats, "/calibration/stats", self.stats_cb, qos_profile_sensor_data)

        self.start_client = n.create_client(StartCollection, "/calibration/start_collection")
        self.stop_client = n.create_client(StopCollection, "/calibration/stop_collection")
        self.clear_client = n.create_client(ClearFrames, "/calibration/clear_frames")

        self.calib_client = n.create_client(CalibrateCamera, "/calibration/calibrate_camera")
        self.stereo_client = n.create_client(CalibrateStereo, "/calibration/calibrate_stereo")

    # ---------- Callbacks ----------
    def image0_cb(self, msg):
        self.last_img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def image1_cb(self, msg):
        self.last_img1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def image0_corr_cb(self, msg):
        self.last_img0_corr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def image1_corr_cb(self, msg):
        self.last_img1_corr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def anno0_cb(self, msg):
        self.last_anno0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def anno1_cb(self, msg):
        self.last_anno1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def tags0_cb(self, msg):
        self.tags0 = msg.tags

    def tags1_cb(self, msg):
        self.tags1 = msg.tags



    def stats_cb(self, msg):
        MAX_FRAMES = 50
        MAX_PAIRS = 50

        # Cam0
        self.cam0_frames_bar.setValue(min(100, int(msg.cam0_frame_count / MAX_FRAMES * 100)))
        self.cam0_frames_label.setText(f"{msg.cam0_frame_count} valid frames")
        self.cam0_div_bar.setValue(int(msg.cam0_diversity * 100))
        self.cam0_cov_bar.setValue(int(msg.cam0_coverage * 100))

        # Cam1
        self.cam1_frames_bar.setValue(min(100, int(msg.cam1_frame_count / MAX_FRAMES * 100)))
        self.cam1_frames_label.setText(f"{msg.cam1_frame_count} valid frames")
        self.cam1_div_bar.setValue(int(msg.cam1_diversity * 100))
        self.cam1_cov_bar.setValue(int(msg.cam1_coverage * 100))

        # Stereo
        self.stereo_frames_bar.setValue(min(100, int(msg.stereo_pair_count / MAX_PAIRS * 100)))
        self.stereo_frames_label.setText(f"{msg.stereo_pair_count} valid stereo pairs")

        # Color border based on ready
        def set_box_color(box, ready):
            color = "#2ecc71" if ready else "#f39c12"  # green / orange
            box.setStyleSheet(f"""
                QGroupBox {{
                    border: 2px solid {color};
                    border-radius: 6px;
                    margin-top: 6px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px 0 4px;
                    color: {color};
                }}
            """)

        set_box_color(self.cam0_box, msg.cam0_ready)
        set_box_color(self.cam1_box, msg.cam1_ready)
        set_box_color(self.stereo_box, msg.stereo_ready)



    # ---------- Overlay ----------
    def draw_tags(self, img, tags):
        if img is None:
            return None
        for tag in tags:
            for i in range(4):
                x = int(tag.corners[2*i])
                y = int(tag.corners[2*i+1])
                cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        return img

    # ---------- Display ----------
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)

    def update_display(self):
        # Cam0
        if hasattr(self, "last_img0") and self.last_img0 is not None:
            pix = self.cv_to_pixmap(self.last_img0)
            pix = pix.scaled(self.cam0_raw_label.width(), self.cam0_raw_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam0_raw_label.setPixmap(pix)

        if hasattr(self, "last_anno0") and self.last_anno0 is not None:
            pix = self.cv_to_pixmap(self.last_anno0)
            pix = pix.scaled(self.cam0_anno_label.width(), self.cam0_anno_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam0_anno_label.setPixmap(pix)

        if hasattr(self, "last_img0_corr") and self.last_img0_corr is not None:
            pix = self.cv_to_pixmap(self.last_img0_corr)
            pix = pix.scaled(self.cam0_corr_label.width(), self.cam0_corr_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam0_corr_label.setPixmap(pix)

        # Cam1
        if hasattr(self, "last_img1") and self.last_img1 is not None:
            pix = self.cv_to_pixmap(self.last_img1)
            pix = pix.scaled(self.cam1_raw_label.width(), self.cam1_raw_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam1_raw_label.setPixmap(pix)

        if hasattr(self, "last_anno1") and self.last_anno1 is not None:
            pix = self.cv_to_pixmap(self.last_anno1)
            pix = pix.scaled(self.cam1_anno_label.width(), self.cam1_anno_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam1_anno_label.setPixmap(pix)

        if hasattr(self, "last_img1_corr") and self.last_img1_corr is not None:
            pix = self.cv_to_pixmap(self.last_img1_corr)
            pix = pix.scaled(self.cam1_corr_label.width(), self.cam1_corr_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam1_corr_label.setPixmap(pix)

    def cv_to_pixmap(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qt_img)


    # ---------- Toggle buttons --------
    def toggle_mono0_collection(self):
        if not self.collecting_mono0:
            # --- Démarrer la collecte ---
            req = StartCollection.Request()
            req.mode = 1  # MODE_MONO_CAM0
            self.start_client.call_async(req)

            # Mettre le flag et changer le texte du bouton
            self.collecting_mono0 = True
            self.btn_start_mono0.setText("Stop Mono 0")
            self.log(f"Start collection: {req.mode}")            
        else:
            # --- Stop la collecte ---
            req = StopCollection.Request()
            req.mode = 1
            self.stop_client.call_async(req)

            # Réinitialiser le flag et le texte du bouton
            self.collecting_mono0 = False
            self.btn_start_mono0.setText("Start Mono 0")
            self.log(f"Stop collection: {req.mode}")

    def toggle_mono1_collection(self):
        if not self.collecting_mono1:
            # --- Démarrer la collecte ---
            req = StartCollection.Request()
            req.mode = 2  # MODE_MONO_CAM1
            self.start_client.call_async(req)

            # Mettre le flag et changer le texte du bouton
            self.collecting_mono1 = True
            self.btn_start_mono1.setText("Stop Mono 1")
            self.log(f"Start collection: {req.mode}")
        else:
            # --- Stop la collecte ---
            req = StopCollection.Request()
            req.mode = 2 # MODE_MONO_CAM1
            self.stop_client.call_async(req)

            # Réinitialiser le flag et le texte du bouton
            self.collecting_mono1 = False
            self.btn_start_mono1.setText("Start Mono 1")
            self.log(f"Stop collection: {req.mode}")

    def toggle_stereo_collection(self):
        if not self.collecting_stereo:
            # --- Démarrer la collecte ---
            req = StartCollection.Request()
            req.mode = 3  # MODE_STEREO
            self.start_client.call_async(req)

            # Mettre le flag et changer le texte du bouton
            self.collecting_stereo = True
            self.btn_start_stereo.setText("Stop stereo")
            self.log(f"Start collection: {req.mode}")
        else:
            # --- Stop la collecte ---
            req = StopCollection.Request()
            req.mode = 3
            self.stop_client.call_async(req)

            # Réinitialiser le flag et le texte du bouton
            self.collecting_stereo = False
            self.btn_start_stereo.setText("Start stereo")
            self.log(f"Stop collection: {req.mode}")


    # ---------- Services ----------
    def log(self, txt):
        self.log_box.append(txt)

    def clear_frames(self):
        self.clear_client.call_async(ClearFrames.Request())
        self.log("Clear frames and stop frame grabbing")
        req = StartCollection.Request()
        req.mode = 0  # MODE_IDLE
        self.start_client.call_async(req)

    def calibrate_camera(self, cam_id):
        req = CalibrateCamera.Request()
        req.camera_id = cam_id
        future = self.calib_client.call_async(req)

        def done_callback(fut):
            try:
                res = fut.result()
                self.display_mono_calibration(cam_id, res)
                self.log(f"Calibration cam{cam_id} finished: {res.message}")
            except Exception as e:
                self.log(f"Calibration cam{cam_id} failed: {e}")

        future.add_done_callback(done_callback)
        self.log(f"Calibration started for cam {cam_id}")

    def calibrate_stereo(self):
        future = self.stereo_client.call_async(CalibrateStereo.Request())

        def done_callback(fut):
            try:
                res = fut.result()
                self.display_stereo_calibration(res)
                self.log(f"Stereo calibration finished: {res.message}")
            except Exception as e:
                self.log(f"Stereo calibration failed: {e}")

        future.add_done_callback(done_callback)
        self.log("Stereo calibration started")


    def display_mono_calibration(self, cam_id, res):
        if not res.success:
            txt = f"Calibration failed:\n{res.message}"
        else:
            txt = f"Reprojection error: {res.reprojection_error:.3f} px\n"
            txt += f"fx: {res.intrinsics[0]:.3f}\n"
            txt += f"fy: {res.intrinsics[1]:.3f}\n"
            txt += f"cx: {res.intrinsics[2]:.3f}\n"
            txt += f"cy: {res.intrinsics[3]:.3f}\n"
            txt += "Distortion: " + ", ".join([f"{d:.6f}" for d in res.distortion])
        
        if cam_id == 0:
            self.calib_results_cam0_text.setText(txt)
        else:
            self.calib_results_cam1_text.setText(txt)

        if hasattr(res, "calibration_file"):
            self.log("Calibration saved to:" + res.calibration_file)



    def display_stereo_calibration(self, res):
        """
        Affiche le résultat de la calibration stéréo.
        res: réponse ROS2 CalibrateStereoResponse
        """
        if not res.success:
            txt = f"Stereo calibration failed:\n{res.message}"
        else:
            txt = f"Reprojection error: {res.reprojection_error:.3f} px\n"

            # Rotation vector (Rodrigues)
            if hasattr(res, "rotation_euler") and len(res.rotation_euler) == 3:
                txt += f"Rotation (radians):\n"
                txt += f"  rx: {res.rotation_euler[0]:.5f}\n"
                txt += f"  ry: {res.rotation_euler[1]:.5f}\n"
                txt += f"  rz: {res.rotation_euler[2]:.5f}\n"

            # Translation vector
            if hasattr(res, "baseline_vector") and len(res.baseline_vector) == 3:
                txt += f"Translation (meters):\n"
                txt += f"  tx: {res.baseline_vector[0]:.5f}\n"
                txt += f"  ty: {res.baseline_vector[1]:.5f}\n"
                txt += f"  tz: {res.baseline_vector[2]:.5f}\n"

            # Matrice 4x4
            if hasattr(res, "RT_stereo") and len(res.t_cam1_cam0) == 16:
                txt += "\n4x4 Matrix (Cam1->Cam0):\n"
                for i in range(4):
                    row = res.t_cam1_cam0[i*4:(i+1)*4]
                    txt += "  " + " ".join(f"{v: .5f}" for v in row) + "\n"

        self.calib_results_stereo_text.setText(txt)

        # Add optimised mono intrinsics
        if hasattr(res, "intrinsics0"):
            txt = self.calib_results_cam0_text.text()
            txt += "\n====== After Stereo Optim ======\n"            
            txt += f"fx: {res.intrinsics0[0]:.3f}\n"
            txt += f"fy: {res.intrinsics0[1]:.3f}\n"
            txt += f"cx: {res.intrinsics0[2]:.3f}\n"
            txt += f"cy: {res.intrinsics0[3]:.3f}\n"
            txt += "Distortion: " + ", ".join([f"{d:.6f}" for d in res.distortion0])
            self.calib_results_cam0_text.setText(txt)

        if hasattr(res, "intrinsics1"):
            txt = self.calib_results_cam1_text.text()
            txt += "\n====== After Stereo Optim ======\n"
            txt += f"fx: {res.intrinsics1[0]:.3f}\n"
            txt += f"fy: {res.intrinsics1[1]:.3f}\n"
            txt += f"cx: {res.intrinsics1[2]:.3f}\n"
            txt += f"cy: {res.intrinsics1[3]:.3f}\n"
            txt += "Distortion: " + ", ".join([f"{d:.6f}" for d in res.distortion1])
            self.calib_results_cam1_text.setText(txt)

        if hasattr(res, "calibration_file"):
            self.log("Calibration saved to: " +res.calibration_file)

# ---------- Main ----------
def main():
    rclpy.init()
    node = CalibrationGUINode()
    sys.exit(node.app.exec())


if __name__ == "__main__":
    main()

