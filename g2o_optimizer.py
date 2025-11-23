import numpy as np
import g2o

class G2OPoseGraphOptimizer:
    # this class implements a pose-graph optimizer utilizing the g2o library, an open-source
    # graph optimization library. this is called whenever an optimized trajectory based on
    # both odometry and loop closure constraints is desired.
    def __init__(self):
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)
    
    def optimize(self, keyframes, loop_edges, odometry_weight=100.0, loop_weight=50.0, 
                 huber_delta=5.0, num_iterations=10, boost_established_loops=False, adaptive_odometry=False):
        self.optimizer.clear()
        kf_id_to_idx = {kf.id: i for i, kf in enumerate(keyframes)}
        
        # this is here incase optimization is run multiple times. the optimization should be ran with the original measurements.
        # i was previously optimizing each loop closure, and the resulting trajectory was diverging from the ground-truth signifigantly
        original_poses = [kf.pose.copy() for kf in keyframes]
        
        
        for i, kf in enumerate(keyframes):
            v = g2o.VertexSE3()
            v.set_id(i)
            v.set_estimate(self.matrix_to_se3(kf.pose))

            # sets the first keyframe as fixed, to match the ground truth trajectory.
            if i == 0:
                v.set_fixed(True)
            self.optimizer.add_vertex(v)
        
        for i in range(len(keyframes) - 1):
            edge = g2o.EdgeSE3()
            edge.set_vertex(0, self.optimizer.vertex(i))
            edge.set_vertex(1, self.optimizer.vertex(i + 1))
            kf_next = keyframes[i + 1]
            if kf_next.relative_pose_from_prev is not None:
                relative_pose = kf_next.relative_pose_from_prev
            else:
                relative_pose = np.linalg.inv(keyframes[i].pose) @ keyframes[i+1].pose
            edge.set_measurement(self.matrix_to_se3(relative_pose))
            
            if adaptive_odometry:
                position_factor = 1.0 - (i / max(len(keyframes) - 1, 1))
                weight_scale = 0.4 + 0.6 * (1.0 - position_factor)
                odometry_weight *= weight_scale
            
            edge.set_information(np.eye(6) * odometry_weight)
            self.optimizer.add_edge(edge)

        for kf1_id, kf2_id, relative_pose in loop_edges:
            edge = g2o.EdgeSE3()
            edge.set_vertex(0, self.optimizer.vertex(kf_id_to_idx[kf1_id]))
            edge.set_vertex(1, self.optimizer.vertex(kf_id_to_idx[kf2_id]))
            edge.set_measurement(self.matrix_to_se3(relative_pose))
            
            kf1 = keyframes[kf_id_to_idx[kf1_id]]
            kf2 = keyframes[kf_id_to_idx[kf2_id]]
            
            loop_info = loop_weight
            
            if boost_established_loops and kf1.num_loop_closures > 0 and kf2.num_loop_closures > 0:
                loop_info *= 1.5
            
            edge.set_information(np.eye(6) * loop_info)
            
            kernel = g2o.RobustKernelHuber()
            kernel.set_delta(huber_delta)
            edge.set_robust_kernel(kernel)
            self.optimizer.add_edge(edge)
        
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(num_iterations)
        
        optimized_poses = []
        optimized_trajectory = []
        for i, kf in enumerate(keyframes):
            v = self.optimizer.vertex(i)
            pose_matrix = self.se3_to_matrix(v.estimate())
            optimized_poses.append(pose_matrix)
            optimized_trajectory.append(pose_matrix[:3, 3])
        
        return optimized_trajectory, optimized_poses
    
    def restore_keyframe_poses(self, keyframes, original_poses):
        for i, kf in enumerate(keyframes):
            kf.pose = original_poses[i].copy()
            kf.position = original_poses[i][:3, 3].copy()
    
    def matrix_to_se3(self, matrix):
        iso = g2o.Isometry3d(matrix)
        return iso
    
    def se3_to_matrix(self, iso):
        return iso.matrix()