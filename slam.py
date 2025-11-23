import numpy as np
import cv2
import glob
import os

from g2o_optimizer import *
from loop_closure import *

class SLAM:
    def __init__(self, sequence='00'):
        # loading data
        self.sequence = sequence
        self.image_path_left = f'./kitti/sequences/{self.sequence}/image_0'
        self.image_path_right = f'./kitti/sequences/{self.sequence}/image_1'
        self.left_images = sorted(glob.glob(os.path.join(self.image_path_left, '*.png')))
        self.right_images = sorted(glob.glob(os.path.join(self.image_path_right, '*.png')))

        self.num_frames = len(self.left_images)
        print(f"Number of frames: {self.num_frames}")

        self.calib_path = './kitti/sequences/' + str(self.sequence) + '/calib.txt'  
        matrix = []
        with open(self.calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('P0:') or line.startswith('P1:'):
                    values = line.split()[1:]
                    matrix.append(np.array(values, dtype=np.float32).reshape(3, 4))

        # camera projection matrices
        self.P0 = matrix[0]
        self.P1 = matrix[1]

        # camera intrinsics
        self.cam0_intrinsics = self.P0[:3, :3]
        self.cam1_intrinsics = self.P1[:3, :3]

        self.cx = self.cam0_intrinsics[0, 2]
        self.cy = self.cam0_intrinsics[1, 2]
        self.fx = self.cam0_intrinsics[0, 0]
        self.fy = self.cam0_intrinsics[1, 1]

        self.baseline = abs(self.P0[0, 3] - self.P1[0, 3]) / self.P0[0, 0]

        self.curr_pose_transform = np.eye(4)

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        self.use_wls_filter = False
        try:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(1.5)
            self.use_wls_filter = True
            print("WLS filtering enabled (opencv-contrib-python installed)")
        except AttributeError:
            print("WLS filtering not available (install opencv-contrib-python for better accuracy)")
            print("Falling back to standard SGBM disparity computation")
        
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.orb = cv2.ORB_create(
            nfeatures=4000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        self.trajectory = [self.curr_pose_transform[:3, 3].copy()]

        # pointcloud storage
        self.all_3d_points = []
        self.all_colors = []
        
        # store pointcloud for each individual frame
        self.keyframe_local_points = []

        self.keyframes = []

        self.loop_detector = LoopClosureDetector(
            self.feature_matcher,
            min_matches=100,
            temporal_gap=50,
            use_bow=True,
            use_spatial=True,
            bow_min_score=0.2,
            spatial_radius=75.0
        )

        self.loop_edges = []
        
        self.bow_training_threshold = 50
        self.bow_trained = False

    def is_valid_depth(self, z, disparity):
        return (z > 0.3 and z < 100.0 and
                disparity > 0.5 and
                disparity < 128.0)

    def compute_filtered_disparity(self, left_image, right_image):
        if self.use_wls_filter:
            disp_left = self.matcher.compute(left_image, right_image)
            disp_right = self.right_matcher.compute(right_image, left_image)
            
            disp_filtered = self.wls_filter.filter(disp_left, left_image, None, disp_right)
            
            disp_filtered = disp_filtered.astype(np.float32) / 16.0
        else:
            disp_filtered = self.matcher.compute(left_image, right_image).astype(np.float32) / 16.0
        
        return disp_filtered

    def get_matches(self, desc1, desc2):
        matches = self.feature_matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        return good_matches
    
    def filter_good_features(self, keypoints, descriptors):
        if len(keypoints) == 0:
            return keypoints, descriptors
            
        good_kps = []
        good_descs = []
        
        for kp, desc in zip(keypoints, descriptors):
            if kp.response > 0.0001:
                good_kps.append(kp)
                good_descs.append(desc)
        
        if len(good_descs) == 0:
            return keypoints, descriptors
            
        return good_kps, np.array(good_descs)
    
    def get_points_3d_and_2d(self, left_points, left_next_points, depth_map, disparity_map):
        points_3d = []
        points_2d_next = []

        for idx, (u, v) in enumerate(left_points):
            u_int, v_int = int(u), int(v)
            
            if (u_int < 2 or v_int < 2 or 
                u_int >= disparity_map.shape[1] - 2 or 
                v_int >= disparity_map.shape[0] - 2):
                continue
            
            disparity = disparity_map[v_int, u_int]
            z = depth_map[v_int, u_int]
            
            if not self.is_valid_depth(z, disparity):
                continue
            
            window = disparity_map[v_int-2:v_int+3, u_int-2:u_int+3]
            if np.std(window) > 3.0:
                continue
            
            if np.sum(window <= 0) > 10:
                continue
            
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            points_3d.append([x, y, z])
            points_2d_next.append(left_next_points[idx])

        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d_next = np.array(points_2d_next, dtype=np.float32)

        return points_3d, points_2d_next


    def perform_PNP(self, points_3d, points_2d_next, cam_intrinsics, frame_id):
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d_next,
            cam_intrinsics,
            None,
            reprojectionError=1.0,
            confidence=0.99,
            iterationsCount=300,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 10:
            print(f"Frame {frame_id}: PnP failed, skipping...")
            self.trajectory.append(self.trajectory[-1])
            return None, None

        if len(inliers) >= 20:
            success, rvec, tvec = cv2.solvePnP(
                points_3d[inliers],
                points_2d_next[inliers],
                cam_intrinsics,
                None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        R, _ = cv2.Rodrigues(rvec)

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()

        return pose, inliers
    
    def valid_keyframe(self, last_keyframe, curr_pose_transform, 
                      translation_threshold=1.0, rotation_threshold=5.0):
        delta_transform = np.linalg.inv(last_keyframe.pose) @ curr_pose_transform
        translation = delta_transform[:3, 3]
        rotation_matrix = delta_transform[:3, :3]
        angle, _ = cv2.Rodrigues(rotation_matrix)
        rotation_angle = np.linalg.norm(angle) * (180.0 / np.pi)
        
        if np.linalg.norm(translation) > translation_threshold or rotation_angle > rotation_threshold:
            return True
        return False
    
    def extract_keyframe_pointcloud(self, keyframe_idx, left_image, right_image, max_points=5000):
        disp_map = self.compute_filtered_disparity(left_image, right_image)
        depth_map = (self.cam0_intrinsics[0, 0] * self.baseline) / (disp_map + 1e-6)
        
        height, width = left_image.shape
        
        step = max(1, int(np.sqrt(height * width / max_points)))
        
        local_points = []
        colors = []
        
        for v in range(0, height, step):
            for u in range(0, width, step):
                disparity = disp_map[v, u]
                z = depth_map[v, u]
                
                if not self.is_valid_depth(z, disparity):
                    continue
                
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                
                local_points.append([x, y, z])
                
                color_val = left_image[v, u] / 255.0
                colors.append([color_val, color_val, color_val])
        
        return np.array(local_points, dtype=np.float32), np.array(colors, dtype=np.float32)
    
    def build_optimized_pointcloud(self, optimized_poses, output_filename, max_points_per_keyframe=5000):
        print(f"\n=== Building optimized point cloud ===")
        print(f"Processing {len(self.keyframe_local_points)} keyframes...")
        
        all_world_points = []
        all_colors = []
        
        for kf_id, local_points, colors in self.keyframe_local_points:
            kf_idx = next((i for i, kf in enumerate(self.keyframes) if kf.id == kf_id), None)
            if kf_idx is None:
                print(f"Warning: Could not find keyframe with id {kf_id}")
                continue
            
            optimized_pose = optimized_poses[kf_idx]

            local_points_homo = np.hstack([local_points, np.ones((len(local_points), 1), dtype=np.float32)])
            
            world_points_homo = (optimized_pose @ local_points_homo.T).T
            
            world_points = world_points_homo[:, :3]
            
            all_world_points.append(world_points)
            all_colors.append(colors)
        
        if len(all_world_points) == 0:
            print("No points to save!")
            return
        
        all_world_points = np.vstack(all_world_points)
        all_colors = np.vstack(all_colors)
        
        print(f"Total points in cloud: {len(all_world_points)}")
        print(f"Point cloud bounds:")
        print(f"  X: [{np.min(all_world_points[:, 0]):.2f}, {np.max(all_world_points[:, 0]):.2f}]")
        print(f"  Y: [{np.min(all_world_points[:, 1]):.2f}, {np.max(all_world_points[:, 1]):.2f}]")
        print(f"  Z: [{np.min(all_world_points[:, 2]):.2f}, {np.max(all_world_points[:, 2]):.2f}]")
        
        if len(self.keyframe_local_points) > 0:
            first_kf_id = self.keyframe_local_points[0][0]
            last_kf_id = self.keyframe_local_points[-1][0]
            print(f"Keyframe ID range: {first_kf_id} to {last_kf_id}")
            print(f"Number of keyframes with point clouds: {len(self.keyframe_local_points)}")
        
        if output_filename.endswith('.ply'):
            self.save_pointcloud_ply(all_world_points, all_colors, output_filename)
        else:
            np.save(output_filename, all_world_points)
            np.save(output_filename.replace('.npy', '_colors.npy'), all_colors)
            print(f"Saved point cloud to {output_filename}")
    
    def save_pointcloud_ply(self, points, colors, filename):
        colors_255 = (colors * 255).astype(np.uint8)
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{colors_255[i, 0]} {colors_255[i, 1]} {colors_255[i, 2]}\n")
        
        print(f"Saved PLY point cloud to {filename}")
    
    def compute_loop_transform(self, current_kf, loop_kf):
        matches = self.get_matches(current_kf.descriptors, loop_kf.descriptors)
        
        if len(matches) < 30:
            print(f"    Not enough matches for loop transform: {len(matches)}")
            return None

        right_image_loop = cv2.imread(self.right_images[loop_kf.id], cv2.IMREAD_GRAYSCALE)
        left_image_loop = loop_kf.left_image

        disp_map_loop = self.compute_filtered_disparity(left_image_loop, right_image_loop)
        depth_map_loop = (self.cam0_intrinsics[0, 0] * self.baseline) / (disp_map_loop + 1e-6)

        points_3d_loop = []
        points_2d_current = []
        
        for match in matches:
            loop_kp = loop_kf.keypoints[match.trainIdx]
            current_kp = current_kf.keypoints[match.queryIdx]
            
            u_loop, v_loop = loop_kp.pt
            u_int, v_int = int(u_loop), int(v_loop)

            if (u_int < 0 or v_int < 0 or 
                u_int >= disp_map_loop.shape[1] or 
                v_int >= disp_map_loop.shape[0]):
                continue
            
            disparity = disp_map_loop[v_int, u_int]
            z = depth_map_loop[v_int, u_int]

            if not self.is_valid_depth(z, disparity):
                continue

            x = (u_loop - self.cx) * z / self.fx
            y = (v_loop - self.cy) * z / self.fy
            points_3d_loop.append([x, y, z])
            points_2d_current.append(current_kp.pt)
        
        if len(points_3d_loop) < 20:
            print(f"    Not enough valid 3D points for loop transform: {len(points_3d_loop)}")
            return None
        
        points_3d_loop = np.array(points_3d_loop, dtype=np.float32)
        points_2d_current = np.array(points_2d_current, dtype=np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d_loop,
            points_2d_current,
            self.cam0_intrinsics,
            None,
            reprojectionError=1.0,
            confidence=0.99,
            iterationsCount=500,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success or inliers is None or len(inliers) < 10:
            print(f"    PnP failed for loop closure")
            return None

        if len(inliers) > 10:
            success, rvec, tvec = cv2.solvePnP(
                points_3d_loop[inliers],
                points_2d_current[inliers],
                self.cam0_intrinsics,
                None,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        R, _ = cv2.Rodrigues(rvec)
        T_loopCam_to_currentCam = np.eye(4)
        T_loopCam_to_currentCam[:3, :3] = R
        T_loopCam_to_currentCam[:3, 3] = tvec.flatten()

        T_loop_to_current_measured = np.linalg.inv(T_loopCam_to_currentCam)
        
        print(f"    Loop transform computed with {len(inliers)} inliers")
        return T_loop_to_current_measured