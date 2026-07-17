"""
VAPE: Viewpoint-Aware Pose Estimation for UAVs
================================================
Reference implementation accompanying the paper:
  "Viewpoint-Aware Template Matching for UAV Relative Pose Estimation"

Real-time 6-DoF relative pose estimation pipeline:
  YOLOv8 detection -> SuperPoint/LightGlue matching against viewpoint
  templates -> coverage-gated PnP-RANSAC (+VVS) -> timestamp-aware UKF.

This file is the reference implementation of the method described in the
paper. Experiment scripts, evaluation tools, and datasets are available
from the corresponding author upon reasonable request.

Usage:
  python vape.py --video_file input.mp4 --save_output
  python vape.py --webcam --show
  (see --help for all options; template images are loaded from
   --template_dir, YOLO weights from --yolo_model)
"""
import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import json
import threading
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import queue
import math
from collections import deque


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE Pose Estimator (Enhanced with Timestamp Support)")
try:
    from ultralytics import YOLO
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from scipy.spatial import cKDTree
    print("✅ All libraries loaded successfully.")
except ImportError as e:
    print(f"❌ Import error: {e}. Please run 'pip install -r requirements.txt' to install dependencies.")
    exit(1)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm > 1e-10:
        return q / norm
    else:

        return np.array([0.0, 0.0, 0.0, 1.0])

def quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_inv(q):
    qn = normalize_quaternion(q)
    return quat_conj(qn)

def quat_to_axis_angle(q):
    qn = normalize_quaternion(q)
    w = float(np.clip(qn[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    axis = np.array([1.0, 0.0, 0.0]) if s < 1e-8 else qn[:3]/s
    return axis, angle

def axis_angle_to_quat(axis, angle):
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle/2.0)
    return normalize_quaternion(np.array([ax[0]*s, ax[1]*s, ax[2]*s, np.cos(angle/2.0)]))

def clamp_quaternion_towards(q_from, q_to, max_deg_per_s, dt):

    if np.dot(q_from, q_to) < 0.0:
        q_to = -q_to


    dq = quat_mul(quat_inv(q_from), q_to)
    axis, ang = quat_to_axis_angle(dq)


    ang_limit = np.deg2rad(max_deg_per_s) * max(dt, 1e-6)

    if ang > ang_limit:

        dq = axis_angle_to_quat(axis, ang_limit)
        q_out = quat_mul(q_from, dq)
    else:
        q_out = q_to

    return normalize_quaternion(q_out)

def clamp_position_towards(pos_from, pos_to, max_speed_m_per_s, dt):

    delta_pos = pos_to - pos_from
    distance = np.linalg.norm(delta_pos)


    max_distance = max_speed_m_per_s * max(dt, 1e-6)

    if distance > max_distance:

        direction = delta_pos / distance
        pos_out = pos_from + direction * max_distance
    else:
        pos_out = pos_to

    return pos_out


@dataclass
class ProcessingResult:
    frame_id: int
    frame: np.ndarray
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    num_inliers: int = 0
    pose_success: bool = False
    viewpoint_used: Optional[str] = None
    capture_time: Optional[float] = None

@dataclass
class PoseData:
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str
    total_matches: int
    coverage_score: float = 0.0


class UnscentedKalmanFilter:
    def __init__(self, dt=1/15.0):
        self.dt = dt
        self.initialized = False


        self.max_rot_rate_dps = 30.0
        self.max_pos_speed_mps = 1.5


        self.n = 16
        self.m = 7


        self.x = np.zeros(self.n)
        self.x[12] = 1.0


        self.P = np.eye(self.n) * 0.1
        self.P0 = self.P.copy()


        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n


        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lambda_)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha**2 + self.beta)


        self.Q = np.eye(self.n) * 1e-1
        self.R = np.eye(self.m) * 1e-4


        self.t_state = None
        self.history = deque(maxlen=200)
        self.min_dt = 1e-6

    def reset(self):
        self.x[:] = 0.0
        self.x[12] = 1.0
        self.P[:] = self.P0
        self.t_state = None
        self.history.clear()
        self.initialized = False

    def _push_history(self):
        if self.t_state is not None:
            self.history.append((self.t_state, self.x.copy(), self.P.copy()))

    def _generate_sigma_points(self, x, P):
        sigmas = np.zeros((2 * self.n + 1, self.n))


        P_sym = 0.5 * (P + P.T)

        try:
            U = np.linalg.cholesky((self.n + self.lambda_) * P_sym)
        except np.linalg.LinAlgError:

            P_jitter = P_sym + 1e-9 * np.eye(self.n)
            try:
                U = np.linalg.cholesky((self.n + self.lambda_) * P_jitter)
            except np.linalg.LinAlgError:

                print("⚠️ Cholesky failed, using SVD fallback")
                U_svd, s, _ = np.linalg.svd(P_jitter)
                U = U_svd @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ U_svd.T
                U = U * np.sqrt(self.n + self.lambda_)

        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1] = x + U[:, i]
            sigmas[self.n+i+1] = x - U[:, i]
        return sigmas

    def motion_model(self, x_in, dt):
        x_out = np.zeros_like(x_in)

        pos, vel, acc = x_in[0:3], x_in[3:6], x_in[6:9]
        x_out[0:3] = pos + vel * dt + 0.5 * acc * dt**2
        x_out[3:6] = vel + acc * dt
        x_out[6:9] = acc

        q, w = x_in[9:13], x_in[13:16]


        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        omega_mat = 0.5 * np.array([
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw],
            [-qx, -qy, -qz]
        ])


        q_dot = omega_mat @ w
        q_new = q + dt * q_dot


        x_out[9:13] = normalize_quaternion(q_new)
        x_out[13:16] = w

        return x_out

    def predict(self, dt):
        if not self.initialized:
            return self.x[0:3], self.x[9:13]
        if dt <= self.min_dt:
            return self.x[0:3], self.x[9:13]

        x_prev = self.x.copy()


        sigmas = self._generate_sigma_points(self.x, self.P)


        sigmas_f = np.array([self.motion_model(s, dt) for s in sigmas])


        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)

        x_pred[9:13] = normalize_quaternion(x_pred[9:13])


        dt_c = max(dt, self.min_dt)
        Q_scaled = self.Q * (dt_c + 0.5 * dt_c * dt_c)

        P_pred = Q_scaled.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.wc[i] * np.outer(y, y)

        P_pred = 0.5 * (P_pred + P_pred.T)


        x_pred[0:3] = clamp_position_towards(
            x_prev[0:3], x_pred[0:3], self.max_pos_speed_mps, dt
        )
        x_pred[9:13] = clamp_quaternion_towards(
            x_prev[9:13], x_pred[9:13], self.max_rot_rate_dps, dt
        )

        self.x = x_pred
        self.P = P_pred

        return self.x[0:3], self.x[9:13]

    def hx(self, x_in):
        z = np.zeros(self.m)
        z[0:3] = x_in[0:3]
        z[3:7] = normalize_quaternion(x_in[9:13])
        return z

    def _measurement_update(self, z_pos, z_quat, R):

        if self.initialized and np.dot(self.x[9:13], z_quat) < 0.0:
            z_quat = -z_quat

        measurement = np.concatenate([z_pos, normalize_quaternion(z_quat)])


        sigmas_f = self._generate_sigma_points(self.x, self.P)


        sigmas_h = np.array([self.hx(s) for s in sigmas_f])


        z_pred = np.sum(self.wm[:, np.newaxis] * sigmas_h, axis=0)

        z_pred[3:7] = normalize_quaternion(z_pred[3:7])

        S = R.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_h[i] - z_pred
            S += self.wc[i] * np.outer(y, y)


        S = 0.5 * (S + S.T)


        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            y_x = sigmas_f[i] - self.x
            y_z = sigmas_h[i] - z_pred
            P_xz += self.wc[i] * np.outer(y_x, y_z)


        try:
            K = P_xz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:

            K = P_xz @ np.linalg.pinv(S)

        innovation = measurement - z_pred
        self.x += K @ innovation


        self.x[9:13] = normalize_quaternion(self.x[9:13])


        self.P -= K @ S @ K.T


        self.P = 0.5 * (self.P + self.P.T)


        min_eigenval = np.min(np.real(np.linalg.eigvals(self.P)))
        if min_eigenval < 1e-12:
            self.P += (1e-9 - min_eigenval) * np.eye(self.n)

    def update_with_timestamp(self, z_pos, z_quat, t_meas, R=None, t_now=None):
        if R is None:
            R = self.R.copy()


        if self.t_state is None:
            self.x[0:3] = z_pos
            self.x[9:13] = normalize_quaternion(z_quat)
            self.P[:] = self.P0
            self.history.clear()
            self.t_state = t_meas
            self.initialized = True
            if t_now is not None and t_now > self.t_state + self.min_dt:
                self._push_history()
                self.predict(t_now - self.t_state)
                self.t_state = t_now
            return self.x[0:3], self.x[9:13]


        if t_meas >= self.t_state:
            dt1 = t_meas - self.t_state
            if dt1 > self.min_dt:
                self._push_history()
                self.predict(dt1)
            self.t_state = t_meas

            self._measurement_update(z_pos, normalize_quaternion(z_quat), R)


            if t_now is not None and t_now > self.t_state + self.min_dt:
                dt2 = t_now - t_meas
                self._push_history()
                self.predict(dt2)
                self.t_state = t_now

            return self.x[0:3], self.x[9:13]


        valid_history = [(i, t, x, P) for i, (t, x, P) in enumerate(self.history) if t <= t_meas]

        if not valid_history:


            return self.x[0:3], self.x[9:13]


        original_history = list(self.history)
        k, t_k, x_k, P_k = valid_history[-1]
        self.x[:] = x_k
        self.P[:] = P_k
        self.t_state = t_k

        if t_meas > t_k + self.min_dt:
            self.predict(t_meas - t_k)
            self.t_state = t_meas

        self._measurement_update(z_pos, normalize_quaternion(z_quat), R)


        replay_history = original_history[:k + 1]
        for j in range(k + 1, len(original_history)):
            t_next = original_history[j][0]
            dt_replay = t_next - self.t_state
            if dt_replay > self.min_dt:
                replay_history.append((self.t_state, self.x.copy(), self.P.copy()))
                self.predict(dt_replay)
                self.t_state = t_next

        if t_now is not None and t_now > self.t_state + self.min_dt:
            replay_history.append((self.t_state, self.x.copy(), self.P.copy()))
            self.predict(t_now - self.t_state)
            self.t_state = t_now

        self.history = deque(replay_history[-self.history.maxlen:], maxlen=self.history.maxlen)
        return self.x[0:3], self.x[9:13]

    def predict_to_time(self, t_target):
        if self.t_state is None or not self.initialized:
            return None, None

        dt = t_target - self.t_state
        if dt <= 0:
            return self.x[0:3], self.x[9:13]


        x_temp = self.x.copy()
        P_temp = self.P.copy()


        sigmas = self._generate_sigma_points(x_temp, P_temp)
        sigmas_f = np.array([self.motion_model(s, dt) for s in sigmas])
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)


        x_pred[9:13] = normalize_quaternion(x_pred[9:13])

        return x_pred[0:3], x_pred[9:13]


    def update(self, position: np.ndarray, quaternion: np.ndarray):
        t_now = time.monotonic()
        return self.update_with_timestamp(position, quaternion, t_now, t_now=t_now)

    def set_rate_limits(self, max_rotation_dps: float = None, max_position_mps: float = None):
        if max_rotation_dps is not None:
            self.max_rot_rate_dps = max_rotation_dps
            print(f"🔄 Rotation rate limit set to {max_rotation_dps}°/s")

        if max_position_mps is not None:
            self.max_pos_speed_mps = max_position_mps
            print(f"🎯 Position speed limit set to {max_position_mps} m/s")

    def set_rotation_rate_limit(self, max_degrees_per_second: float):
        self.set_rate_limits(max_rotation_dps=max_degrees_per_second)


class MainThread(threading.Thread):
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.camera_width, self.camera_height = 1280, 720
        self.is_video_stream = False
        self.video_capture = None
        self.image_files = []
        self.frame_idx = 0
        self.frame_count = 0
        self.start_time = time.monotonic()

        self._initialize_input_source()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_input_source(self):
        if self.args.webcam:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened(): raise IOError("Cannot open webcam.")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.is_video_stream = True
            print("📹 Using webcam input.")
        elif self.args.video_file:
            if not os.path.exists(self.args.video_file): raise FileNotFoundError(f"Video file not found: {self.args.video_file}")
            self.video_capture = cv2.VideoCapture(self.args.video_file)
            self.is_video_stream = True
            print(f"📹 Using video file input: {self.args.video_file}")
        elif self.args.image_dir:
            if not os.path.exists(self.args.image_dir): raise FileNotFoundError(f"Image directory not found: {self.args.image_dir}")
            self.image_files = sorted([os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not self.image_files: raise IOError(f"No images found in directory: {self.args.image_dir}")
            print(f"🖼️ Found {len(self.image_files)} images for processing.")
        else:
            raise ValueError("No input source specified. Use --webcam, --video_file, or --image_dir.")

    def _get_next_frame(self):
        if self.is_video_stream:
            ret, frame = self.video_capture.read()
            return frame if ret else None
        else:
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                self.frame_idx += 1
                return frame
            return None

    def run(self):
        window_name = "VAPE - Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height)

        while self.running:
            loop_start_time = time.monotonic()


            frame = self._get_next_frame()
            if frame is None:
                break

            t_capture = time.monotonic()


            t_now = time.monotonic()
            with self.pose_data_lock:

                predicted_pose_tvec, predicted_pose_quat = self.kf.predict_to_time(t_now)


            vis_frame = frame.copy()
            if predicted_pose_tvec is not None and predicted_pose_quat is not None:
                self._draw_axes(vis_frame, predicted_pose_tvec, predicted_pose_quat)


            elapsed_time = time.monotonic() - self.start_time
            fps = (self.frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_frame, "STATUS: PREDICTING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)


            with self.pose_data_lock:
                filter_time = self.kf.t_state
            if filter_time is not None:
                age_ms = (t_now - filter_time) * 1000
                cv2.putText(vis_frame, f"Filter Age: {age_ms:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            self.frame_count += 1


            if self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()
                except queue.Empty:
                    pass
            self.processing_queue.put((frame.copy(), t_capture))


            if self.args.show:
                try:
                    vis_data = self.visualization_queue.get_nowait()
                    kpts, vis_crop = vis_data['kpts'], vis_data['crop']
                    for kpt in kpts:
                        cv2.circle(vis_crop, (int(kpt[0]), int(kpt[1])), 2, (0, 255, 0), -1)
                    cv2.imshow("SuperPoint Features", vis_crop)
                except queue.Empty:
                    pass


            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                print("User requested shutdown.")
                break


            frame_rate_cap = 30.0
            time_to_wait = (1.0 / frame_rate_cap) - (time.monotonic() - loop_start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        self.cleanup()

    # def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
    #     fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
    #     K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    #     return K, None

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix K and distortion coefficients."""
        fx, fy, cx, cy = 1078.86998, 1074.77105, 640.626268, 377.596433
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([
            0.02692405, -0.03433880, 0.01104186, 0.00124234, -0.12498783
        ], dtype=np.float32)
        return K, dist_coeffs


    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        q_norm = normalize_quaternion(q)
        x, y, z, w = q_norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)
        except (cv2.error, AttributeError, ValueError):
            pass

    def cleanup(self):
        self.running = False
        if self.is_video_stream and self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()


class ProcessingThread(threading.Thread):
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_width, self.camera_height = 1280, 720
        self.all_poses_log = []


        self.current_best_viewpoint = None
        self.needs_full_eval = True
        self.theta_reuse = 15.0


        self.COVERAGE_ACCEPT = 0.55
        self.COVERAGE_MIN = 0.25
        self.MIN_INLIERS_MIDBAND = 6


        self.last_orientation: Optional[np.ndarray] = None
        self.ORI_MAX_DIFF_DEG = 30
        self.rejected_consecutive_frames_count = 0
        self.MAX_REJECTED_FRAMES = 5
        self.MAX_KF_PREDICTION_AGE_S = 1.0

        self.yolo_model = None
        self.extractor = None
        self.matcher = None
        self.viewpoint_anchors = {}

        self._initialize_models()
        self._initialize_anchor_data()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_models(self):
        print("📦 Loading models...")
        self.yolo_model = YOLO(self.args.yolo_model).to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024, detection_threshold=0.03).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("   ...models loaded.")

    def _initialize_anchor_data(self):
        print("🛠️ Initializing anchor data...")


        ne_anchor_2d = np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],[351, 228],[338, 220],[294, 244],[230, 541],[401, 469],[414, 481],[464, 451],[521, 510],[610, 454],[544, 400],[589, 373],[575, 361],[486, 561],[739, 385],[826, 305],[791, 285],[773, 271],[845, 233],[826, 226],[699, 308],[790, 375]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],[-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, 0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)
        sw_d_anchor_2d = np.array([[947, 319],[394, 185],[343,476 ],[509,354 ],[241,261]], dtype=np.float32)
        sw_d_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113]], dtype=np.float32)
        se_d_anchor_2d = np.array([[829, 139],[191, 424],[821,570 ],[935,316 ],[602,513 ]], dtype=np.float32)
        se_d_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113]], dtype=np.float32)
        nw_d_anchor_2d = np.array([[280,421 ],[823,161 ],[420,202 ],[305,158 ],[532,72 ]], dtype=np.float32)
        nw_d_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113]], dtype=np.float32)
        ne_d_anchor_2d = np.array([[355,242 ],[854,558 ],[763,260 ],[644,142 ],[867,238 ]], dtype=np.float32)
        ne_d_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113]], dtype=np.float32)



        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'path': 'NW.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
            'SW': {'path': 'SW.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d},
            'SW_d': {'path': 'SW_d.png', '2d': sw_d_anchor_2d, '3d': sw_d_anchor_3d},
            'SE_d': {'path': 'SE_d.png', '2d': se_d_anchor_2d, '3d': se_d_anchor_3d},
            'NW_d': {'path': 'NW_d.png', '2d': nw_d_anchor_2d, '3d': nw_d_anchor_3d},
            'NE_d': {'path': 'NE_d.png', '2d': ne_d_anchor_2d, '3d': ne_d_anchor_3d},
        }

        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            if not os.path.exists(data['path']):
                raise FileNotFoundError(f"Required anchor image not found: {data['path']}")

            anchor_image_bgr = cv2.resize(cv2.imread(os.path.join(self.args.template_dir, data['path'])), (self.camera_width, self.camera_height))
            anchor_features = self._extract_features_sp(anchor_image_bgr)
            anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()

            sp_tree = cKDTree(anchor_keypoints)
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
            valid_mask = distances != np.inf

            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
            }
        print("   ...anchor data initialized.")

    def run(self):
        frame_id = 0
        while self.running:
            try:
                frame_data = self.processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue


            while True:
                try:
                    frame_data = self.processing_queue.get_nowait()
                except queue.Empty:
                    break


            if isinstance(frame_data, tuple) and len(frame_data) == 2:
                frame, t_capture = frame_data
            else:

                frame = frame_data
                t_capture = time.monotonic()

            result = self._process_frame(frame, frame_id, t_capture)
            frame_id += 1


            self.all_poses_log.append({
                'frame': result.frame_id,
                'success': result.pose_success,
                'position': result.position.tolist() if result.position is not None else None,
                'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
                'num_inliers': result.num_inliers,
                'viewpoint_used': result.viewpoint_used,
                'capture_time': t_capture
            })

    def _process_frame(self, frame: np.ndarray, frame_id: int, t_capture: float) -> ProcessingResult:
        result = ProcessingResult(frame_id=frame_id, frame=frame.copy(), pose_success=False, capture_time=t_capture)


        bbox = self._yolo_detect(frame)
        result.bbox = bbox


        best_pose = self._estimate_pose_with_temporal_consistency(frame, bbox)


        is_valid = False
        if best_pose:
            orientation_valid = True
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, best_pose.quaternion))
                if angle_diff > self.ORI_MAX_DIFF_DEG:
                    orientation_valid = False
                    print(f"🚫 Frame {frame_id}: Rejected (Orientation Jump: {angle_diff:.1f}° > {self.ORI_MAX_DIFF_DEG}°)")


            cov = best_pose.coverage_score
            if cov > self.COVERAGE_ACCEPT:
                coverage_valid = True
            elif cov >= self.COVERAGE_MIN:

                coverage_valid = best_pose.inliers >= self.MIN_INLIERS_MIDBAND
                if not coverage_valid:
                    print(f"🚫 Frame {frame_id}: Rejected (Coverage {cov:.2f} in mid-band, "
                          f"inliers {best_pose.inliers} < {self.MIN_INLIERS_MIDBAND})")
            else:
                coverage_valid = False
                print(f"🚫 Frame {frame_id}: Rejected (Coverage {cov:.2f} < {self.COVERAGE_MIN})")


            is_valid = orientation_valid and coverage_valid

            if is_valid:
                print(f"✅ Frame {frame_id}: Accepted (Orientation: {orientation_valid}, "
                      f"Coverage: {best_pose.coverage_score:.2f}, "
                      f"Inliers: {best_pose.inliers}/{best_pose.total_matches})")


        if is_valid and best_pose:
            self.rejected_consecutive_frames_count = 0
            result.position, result.quaternion = best_pose.position, best_pose.quaternion
            result.num_inliers, result.pose_success = best_pose.inliers, True
            result.viewpoint_used = best_pose.viewpoint
            self.last_orientation = best_pose.quaternion


            base_pos_noise = 1e-4
            base_quat_noise = 1e-4
            gamma = 1.0
            noise_scale = 1.0 + gamma * best_pose.reprojection_error / max(best_pose.inliers, 1)

            R = np.eye(7)
            R[0:3, 0:3] *= base_pos_noise * noise_scale
            R[3:7, 3:7] *= base_quat_noise * noise_scale

            t_now = time.monotonic()

            with self.pose_data_lock:
                kf_pos, kf_quat = self.kf.update_with_timestamp(
                    best_pose.position,
                    best_pose.quaternion,
                    t_meas=t_capture,
                    R=R,
                    t_now=t_now
                )
                result.kf_position, result.kf_quaternion = kf_pos, kf_quat
        else:

            self.rejected_consecutive_frames_count += 1

            should_reset_kf = self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES
            with self.pose_data_lock:
                prediction_age = (
                    time.monotonic() - self.kf.t_state
                    if self.kf.t_state is not None and self.kf.initialized
                    else 0.0
                )
                should_reset_kf = should_reset_kf or prediction_age > self.MAX_KF_PREDICTION_AGE_S

            if should_reset_kf:
                print(f"⚠️ KF tracking lost "
                      f"(rejections={self.rejected_consecutive_frames_count}, "
                      f"prediction_age={prediction_age:.2f}s). Re-initializing KF.")
                with self.pose_data_lock:
                    self.kf.reset()
                self.last_orientation = None
                self.current_best_viewpoint = None
                self.needs_full_eval = True
                self.rejected_consecutive_frames_count = 0

        return result

    def _estimate_pose_with_temporal_consistency(self, frame: np.ndarray, bbox: Optional[Tuple]) -> Optional[PoseData]:
        LAMBDA_INLIERS = 1.0
        LAMBDA_COVERAGE = 15.0

        def compute_score(pose: PoseData) -> float:
            return LAMBDA_INLIERS * pose.inliers + LAMBDA_COVERAGE * pose.coverage_score


        if self.current_best_viewpoint and not self.needs_full_eval:
            pose_data = self._solve_for_viewpoint(frame, self.current_best_viewpoint, bbox)
            if pose_data:
                score = compute_score(pose_data)
                if score > self.theta_reuse:

                    return pose_data

            self.needs_full_eval = True


        all_poses = []
        for viewpoint in self.viewpoint_anchors.keys():
            pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
            if pose_data:
                all_poses.append(pose_data)

        if not all_poses:
            return None


        best_pose = max(all_poses, key=lambda p: compute_score(p))
        best_score = compute_score(best_pose)


        if best_pose.viewpoint == self.current_best_viewpoint and best_score > self.theta_reuse:

            self.needs_full_eval = False
        else:

            self.needs_full_eval = True

        self.current_best_viewpoint = best_pose.viewpoint
        print(f"🎯 Selected viewpoint: {best_pose.viewpoint} "
              f"(inliers={best_pose.inliers}, "
              f"coverage={best_pose.coverage_score:.2f}, "
              f"score={best_score:.1f}, "
              f"reuse={'yes' if not self.needs_full_eval else 'no'})")
        return best_pose

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None

        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0: return None


        frame_features = self._extract_features_sp(crop)


        if self.args.show:
            if self.visualization_queue.full():
                try:
                    self.visualization_queue.get_nowait()
                except queue.Empty:
                    pass
            kpts = frame_features['keypoints'][0].cpu().numpy()
            self.visualization_queue.put({'kpts': kpts, 'crop': crop.copy()})


        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        if len(matches) < 6: return None


        points_3d, points_2d = [], []
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        if len(points_3d) < 6: return None


        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(points_3d, dtype=np.float32),
                np.array(points_2d, dtype=np.float32),
                self.K, self.dist_coeffs, reprojectionError=8, confidence=0.95,
                iterationsCount=3000, flags=cv2.SOLVEPNP_EPNP
            )

            if success and inliers is not None and len(inliers) > 4:
                rvec, tvec = cv2.solvePnPRefineVVS(
                    np.array(points_3d, dtype=np.float32)[inliers.flatten()],
                    np.array(points_2d, dtype=np.float32)[inliers.flatten()],
                    self.K, self.dist_coeffs, rvec, tvec
                )
        except cv2.error as e:
            return None

        if not success or inliers is None or len(inliers) < 4: return None


        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion_raw = self._rotation_matrix_to_quaternion(R)

        quaternion = normalize_quaternion(quaternion_raw)

        projected_points, _ = cv2.projectPoints(np.array(points_3d)[inliers.flatten()], rvec, tvec, self.K, self.dist_coeffs)
        error = np.mean(np.linalg.norm(np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))


        inlier_3d = np.array(points_3d, dtype=np.float32)[inliers.flatten()]
        coverage_score = self._compute_coverage_score(inlier_3d)

        return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d), coverage_score)

    def _compute_coverage_score(self, inlier_3d_points: np.ndarray) -> float:
        if len(inlier_3d_points) == 0:
            return 0.0


        regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}
        for pt in inlier_3d_points:


            if pt[0] < 0 and pt[2] > 0:
                regions["front-right"] += 1
            elif pt[0] < 0 and pt[2] < 0:
                regions["front-left"] += 1
            elif pt[0] >= 0 and pt[2] > 0:
                regions["back-right"] += 1
            elif pt[0] >= 0 and pt[2] <= 0:
                regions["back-left"] += 1

        total_points = sum(regions.values())
        if total_points == 0:
            return 0.0


        entropy_sum = 0.0
        for count in regions.values():
            if count > 0:
                proportion = count / total_points
                entropy_sum += proportion * np.log(proportion)


        coverage_score = -entropy_sum / np.log(4)
        return float(np.clip(coverage_score, 0.0, 1.0))

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        dot = np.dot(normalize_quaternion(q1), normalize_quaternion(q2))
        return 2 * math.acos(abs(min(1.0, max(-1.0, dot))))

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:

        names = getattr(self.yolo_model, "names", {0: "iha"})
        inv = {v: k for k, v in names.items()}
        target_id = inv.get("iha", 0)

        for conf_thresh in (0.90, 0.75, 0.65):
            results = self.yolo_model(
                frame,
                imgsz=640,
                conf=conf_thresh,
                iou=0.5,
                max_det=5,
                classes=[target_id],
                verbose=False
            )
            if not results or len(results[0].boxes) == 0:
                continue


            boxes = results[0].boxes
            best = max(
                boxes,
                key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
            )
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
            return int(x1), int(y1), int(x2), int(y2)

        return None

    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    # def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
    #     fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
    #     K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    #     dist_coeffs = np.array([
    #          0.02692405, -0.03433880, 0.01104186, 0.00124234, -0.12498783
    #      ], dtype=np.float32)
    #     return K, dist_coeffs

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix K and distortion coefficients."""
        fx, fy, cx, cy = 1078.86998, 1074.77105, 640.626268, 377.596433
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([
            0.02692405, -0.03433880, 0.01104186, 0.00124234, -0.12498783
        ], dtype=np.float32)
        return K, dist_coeffs

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

    def cleanup(self):
        print("Shutting down...")
        self.running = False
        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4)
            print(f"Pose log saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="VAPE - Real-time Pose Estimator with Timestamp Support")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    parser.add_argument('--yolo_model', type=str, default='weights/yolo_uav.pt', help='Path to the YOLO detector weights.')
    parser.add_argument('--template_dir', type=str, default='templates', help='Directory containing viewpoint template images.')
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    parser.add_argument('--show', action='store_true', help='Show keypoint detections in a separate window.')
    args = parser.parse_args()

    main_thread = None
    processing_thread = None

    try:

        processing_queue = queue.Queue(maxsize=1)
        visualization_queue = queue.Queue(maxsize=2)
        pose_data_lock = threading.Lock()


        kf = UnscentedKalmanFilter()

        kf.set_rate_limits(max_rotation_dps=30.0, max_position_mps=1.5)

        main_thread = MainThread(processing_queue, visualization_queue, pose_data_lock, kf, args)
        processing_thread = ProcessingThread(processing_queue, visualization_queue, pose_data_lock, kf, args)

        print("Starting VAPE in enhanced multi-threaded mode with timestamp support...")
        main_thread.start()
        processing_thread.start()


        main_thread.join()

    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user (Ctrl+C).")
    finally:
        if main_thread is not None and main_thread.is_alive():
            main_thread.running = False
            main_thread.join(timeout=1.0)

        if processing_thread is not None:
            if processing_thread.is_alive():
                print("Stopping processing thread...")
                processing_thread.running = False
                processing_thread.join(timeout=2.0)
            processing_thread.cleanup()

        print("✅ Process finished.")


if __name__ == '__main__':
    main()
