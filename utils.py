import numpy as np
import cv2
from collections import defaultdict
from scipy.spatial import KDTree

class Keyframe:
    # this class defines what a "keyframe" is, and associated functions which allow the implementation of the SLAM algorithm.
    # each keyframe stores information about itself that allows us to determine loop closures, as well as an uncertainty used
    # in the graph optimization. additionally, each point in the pointcloud from this image is stored so later
    # an optimized pointcloud can be created after the graph is optimized.
    def __init__(self, frame_id, left_image, keypoints, descriptors, pose_transform, relative_pose_from_prev=None):
        self.id = frame_id
        self.pose = pose_transform.copy()
        self.left_image = left_image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_points = {}
        self.connections = {}
        self.relative_pose_from_prev = relative_pose_from_prev
        self.num_loop_closures = 0
        self.uncertainty = 1.0
        
        self.bow_vector = None
        self.position = pose_transform[:3, 3].copy()
    
    def add_map_point(self, keypoint_idx, map_point):
        self.map_points[keypoint_idx] = map_point
        map_point.add_observation(self.id, keypoint_idx)


class BoWDatabase:
    # this class stores the Bag of Words information to allow us to efficiently find keyframe matches for loop closures.
    # a dictionary of "words" is created based upon a number of keyframes, where the "words" describe different attributes
    # of the images. Then, we create a BoW vector for eahc image and store it. when a new keyframe is added, we search the
    # stored vectors for a close match to determine loop closures. this method is signifigantly more efficient than feature
    # matching every image, which is something that I had done previously.
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
        self.vocabulary = None
        self.inverted_index = defaultdict(set)
        self.keyframe_vectors = {}
        self.is_trained = False
    
    def train_vocabulary(self, all_descriptors):
        print(f"Training BoW vocabulary with {len(all_descriptors)} descriptors...")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        _, labels, centers = cv2.kmeans(
            all_descriptors.astype(np.float32),
            self.vocabulary_size,
            None,
            criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS
        )
        
        self.vocabulary = centers.astype(np.uint8)
        self.is_trained = True
        print(f"BoW vocabulary trained with {self.vocabulary_size} words")
    
    def compute_bow_vector(self, descriptors):
        if not self.is_trained:
            return None
        
        bow_vector = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        for desc in descriptors:
            distances = np.sum((self.vocabulary.astype(np.float32) - desc.astype(np.float32)) ** 2, axis=1)
            word_id = np.argmin(distances)
            bow_vector[word_id] += 1.0
        
        if np.sum(bow_vector) > 0:
            bow_vector /= np.sum(bow_vector)
        
        norm = np.linalg.norm(bow_vector)
        if norm > 0:
            bow_vector /= norm
        
        return bow_vector
    
    def add_keyframe(self, keyframe):
        if not self.is_trained:
            return
        
        bow_vector = self.compute_bow_vector(keyframe.descriptors)
        keyframe.bow_vector = bow_vector
        self.keyframe_vectors[keyframe.id] = bow_vector
        
        for word_id, count in enumerate(bow_vector):
            if count > 0:
                self.inverted_index[word_id].add(keyframe.id)
    
    def query(self, keyframe, min_score=0.5):
        if not self.is_trained or keyframe.bow_vector is None:
            return []
        
        candidate_ids = set()
        for word_id, count in enumerate(keyframe.bow_vector):
            if count > 0:
                candidate_ids.update(self.inverted_index[word_id])
        
        scores = []
        for kf_id in candidate_ids:
            if kf_id == keyframe.id:
                continue
            
            candidate_vector = self.keyframe_vectors[kf_id]
            score = np.dot(keyframe.bow_vector, candidate_vector)
            
            if score >= min_score:
                scores.append((kf_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class SpatialIndex:
    # this class builds a KD-tree based upon the 3D positions of the keyframes,
    # allowing for us to efficiently find keyframes near a 3D position.
    def __init__(self):
        self.keyframes = []
        self.positions = None
        self.kdtree = None
    
    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        self._rebuild_tree()
    
    def _rebuild_tree(self):
        if len(self.keyframes) == 0:
            return
        
        self.positions = np.array([kf.position for kf in self.keyframes])
        self.kdtree = KDTree(self.positions)
    
    def query_radius(self, position, radius=20.0):
        if self.kdtree is None:
            return []
        
        indices = self.kdtree.query_ball_point(position, radius)
        return [self.keyframes[i] for i in indices]
    
    def query_k_nearest(self, position, k=10):
        if self.kdtree is None or len(self.keyframes) < k:
            return []
        
        distances, indices = self.kdtree.query(position, k=min(k, len(self.keyframes)))
        
        if not isinstance(indices, np.ndarray):
            indices = [indices]
            distances = [distances]
        
        return [(self.keyframes[i], distances[j]) for j, i in enumerate(indices)]