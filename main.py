import numpy as np
import cv2

from slam import *


slam = SLAM(sequence='05')

left_image = cv2.imread(slam.left_images[0], cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(slam.right_images[0], cv2.IMREAD_GRAYSCALE)

disp_map = slam.compute_filtered_disparity(left_image, right_image)
depth_map = (slam.cam0_intrinsics[0, 0] * slam.baseline) / (disp_map + 1e-6)

left_keypoints, left_descriptors = slam.orb.detectAndCompute(left_image, None)

left_keypoints, left_descriptors = slam.filter_good_features(left_keypoints, left_descriptors)

initial_keyframe = Keyframe(0, left_image, left_keypoints, left_descriptors, 
                            slam.curr_pose_transform.copy())
slam.keyframes.append(initial_keyframe)

optimized_trajectory = None

for i in range(1, slam.num_frames-1):
    # read images
    left_image = cv2.imread(slam.left_images[i], cv2.IMREAD_GRAYSCALE)
    left_image_next = cv2.imread(slam.left_images[i+1], cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(slam.right_images[i], cv2.IMREAD_GRAYSCALE)

    if i % 100 == 0:
        print(f"Processing frame {i}/{slam.num_frames-1}")

    disp_map = slam.compute_filtered_disparity(left_image, right_image)
    depth_map = (slam.cam0_intrinsics[0, 0] * slam.baseline) / (disp_map + 1e-6)

    # keypoint and descriptor extraction from consecutive left frames
    left_keypoints, left_descriptors = slam.orb.detectAndCompute(left_image, None)
    left_next_keypoints, left_next_descriptors = slam.orb.detectAndCompute(left_image_next, None)

    left_keypoints, left_descriptors = slam.filter_good_features(left_keypoints, left_descriptors)
    left_next_keypoints, left_next_descriptors = slam.filter_good_features(left_next_keypoints, left_next_descriptors)

    # match features between temporally in left frames
    good_matches = slam.get_matches(left_descriptors, left_next_descriptors)

    # reconstruct 3D points from depth map
    left_points = np.float32([left_keypoints[m.queryIdx].pt for m in good_matches])
    left_next_points = np.float32([left_next_keypoints[m.trainIdx].pt for m in good_matches])

    points_3d, points_2d_next = slam.get_points_3d_and_2d(left_points, left_next_points, depth_map, disp_map)

    # checks for at least 10 points for PnP (reasonable threshold)
    if len(points_3d) < 10:
        print(f"Frame {i}: Not enough valid 3D points ({len(points_3d)}), skipping...")
        slam.trajectory.append(slam.trajectory[-1])
        continue
    
    # pose estimation via PnP
    pose, inliers = slam.perform_PNP(points_3d, points_2d_next, slam.cam0_intrinsics, i)
    if pose is None:
        continue
    R = pose[:3, :3]
    tvec = pose[:3, 3].reshape(3,1)

    # solvePNP returns transformation from current to next frame
    T_current_to_next = np.eye(4)
    T_current_to_next[:3, :3] = R
    T_current_to_next[:3, 3] = tvec.flatten()
    
    # invert to get next to current
    T_next_to_current = np.linalg.inv(T_current_to_next)
    
    # compute accumulated pose
    slam.curr_pose_transform = slam.curr_pose_transform @ T_next_to_current
    
    # extract and store trajectory point
    new_position = slam.curr_pose_transform[:3, 3]
    slam.trajectory.append(new_position.copy())

    should_keyframe = slam.valid_keyframe(
        slam.keyframes[-1],
        slam.curr_pose_transform,
        translation_threshold=2.5,
        rotation_threshold=5.0
    )

    if should_keyframe:
        # create new keyframe
        relative_pose_from_prev_kf = np.linalg.inv(slam.keyframes[-1].pose) @ slam.curr_pose_transform
        new_keyframe = Keyframe(
            i,
            left_image,
            left_keypoints,
            left_descriptors,
            slam.curr_pose_transform.copy(),
            relative_pose_from_prev=relative_pose_from_prev_kf
        )
        slam.keyframes.append(new_keyframe)
        
        # extract pointcloud from every 5th keyframe, reducing computational load
        if i % 5 == 0:  
            local_points, colors = slam.extract_keyframe_pointcloud(
                len(slam.keyframes) - 1, 
                left_image, 
                right_image,
                max_points=5000
            )
            if len(local_points) > 0:
                slam.keyframe_local_points.append((i, local_points, colors))
                if len(slam.keyframe_local_points) % 10 == 0:
                    print(f"Extracted point cloud for keyframe {i} ({len(local_points)} points)")

        if not slam.bow_trained and len(slam.keyframes) >= slam.bow_training_threshold:
            print(f"=== Training BoW vocabulary with {len(slam.keyframes)} keyframes ===")
            slam.loop_detector.train_bow_vocabulary(slam.keyframes)
            slam.bow_trained = True
            print(f"=== BoW training complete ===")

        if slam.bow_trained:
            slam.loop_detector.add_keyframe(new_keyframe)

        # detect loop closures
        loop_kf = slam.loop_detector.detect(new_keyframe, slam.keyframes[:-1])
        if loop_kf is not None:
            print(f"Loop detected between keyframe {new_keyframe.id} and {loop_kf.id}!")
            
            relative_pose_loop = slam.compute_loop_transform(new_keyframe, loop_kf)
            
            if relative_pose_loop is not None:
                slam.loop_edges.append((loop_kf.id, new_keyframe.id, relative_pose_loop))
                print(f"Loop edge added. Total loop closures: {len(slam.loop_edges)}")
                
                loop_kf.num_loop_closures += 1
                new_keyframe.num_loop_closures += 1
            else:
                print(f"Loop transform computation failed, loop edge not added")


np.save(f"kitti_{slam.sequence}_vo_trajectory.npy", np.array(slam.trajectory))
print(f"Saved VO trajectory with {len(slam.trajectory)} frames")


if len(slam.loop_edges) > 0:
    print(f"Optimizing Trajectory")
    
    # optimizer information. change weights here to adjust how much each type of edge is prioritized
    odometry_weight=100.0
    loop_weight = 50.0
    huber_delta = 5.0
    
    optimizer = G2OPoseGraphOptimizer()

    optimized_trajectory, optimized_poses = optimizer.optimize(
        slam.keyframes, 
        slam.loop_edges,
        odometry_weight=100.0,
        loop_weight=50.0,
        huber_delta=5.0,
        num_iterations=10,
        boost_established_loops=False
    )

    filename = f"kitti_{slam.sequence}_optimized.npy"
    np.save(filename, optimized_trajectory)
    print(f"Saved: {filename}")

    
    pointcloud_filename = f"kitti_{slam.sequence}_pointcloud.ply"
    slam.build_optimized_pointcloud(optimized_poses, pointcloud_filename, max_points_per_keyframe=5000)

else:
    print(f"No loop closures detected - skipping optimization")

    if len(slam.keyframe_local_points) > 0:
        print(f"=== Generating point cloud from VO trajectory ===")
        vo_poses = [kf.pose for kf in slam.keyframes]
        pointcloud_filename = f"kitti_{slam.sequence}_pointcloud_slam.ply"
        slam.build_optimized_pointcloud(vo_poses, pointcloud_filename, max_points_per_keyframe=5000)

print(f"=== SLAM Complete ===")
print(f"Total keyframes: {len(slam.keyframes)}")
print(f"Total loop closures: {len(slam.loop_edges)}")
print(f"Total frames processed: {len(slam.trajectory)}")
