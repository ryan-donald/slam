import numpy as np
from utils import *

class LoopClosureDetector:
    # this class implements a loop closure detector, which creates a BoWDatabase and SpatialIndex object,
    # which both combine to allow for efficient detection of matched keyframes, and as a result efficient
    # detection of loop closures. 

    def __init__(self, feature_matcher, min_matches=100, temporal_gap=30, 
                 use_bow=True, use_spatial=True, bow_min_score=0.3, spatial_radius=20.0):
        self.feature_matcher = feature_matcher
        self.min_matches = min_matches
        self.temporal_gap = temporal_gap
        self.use_bow = use_bow
        self.use_spatial = use_spatial
        self.bow_min_score = bow_min_score
        self.spatial_radius = spatial_radius
        
        self.bow_database = BoWDatabase(vocabulary_size=1000) if use_bow else None
        self.spatial_index = SpatialIndex() if use_spatial else None
    
    def train_bow_vocabulary(self, keyframes):
        if not self.use_bow:
            return
        
        all_descriptors = []
        for kf in keyframes:
            all_descriptors.append(kf.descriptors)
        
        all_descriptors = np.vstack(all_descriptors)
        self.bow_database.train_vocabulary(all_descriptors)
        
        for kf in keyframes:
            self.bow_database.add_keyframe(kf)
            if self.use_spatial:
                self.spatial_index.add_keyframe(kf)
    
    def add_keyframe(self, keyframe):
        if self.use_bow and self.bow_database.is_trained:
            self.bow_database.add_keyframe(keyframe)
        if self.use_spatial:
            self.spatial_index.add_keyframe(keyframe)
    
    def detect(self, current_keyframe, all_keyframes):
        
        if len(all_keyframes) < self.temporal_gap:
            return None
        
        candidates = [kf for kf in all_keyframes 
                     if abs(kf.id - current_keyframe.id) > self.temporal_gap]
        
        if len(candidates) == 0:
            return None
        
        if self.use_bow and self.bow_database.is_trained:
            bow_results = self.bow_database.query(current_keyframe, min_score=self.bow_min_score)
            
            if len(bow_results) == 0:
                return None
            
            bow_candidate_ids = {kf_id for kf_id, _ in bow_results[:20]}
            candidates = [kf for kf in candidates if kf.id in bow_candidate_ids]
        
        if self.use_spatial and len(candidates) > 0:
            spatial_candidates = self.spatial_index.query_radius(
                current_keyframe.position, 
                radius=self.spatial_radius
            )
            spatial_candidate_ids = {kf.id for kf in spatial_candidates}
            
            candidates = [kf for kf in candidates if kf.id in spatial_candidate_ids]
        
        if len(candidates) == 0:
            return None
        
        best_match_score = 0
        best_match_kf = None
        
        for candidate_kf in candidates:
            
            matches = self.feature_matcher.knnMatch(
                current_keyframe.descriptors, 
                candidate_kf.descriptors, 
                k=2
            )
            
            good_matches = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            num_matches = len(good_matches)
            
            if num_matches > best_match_score:
                best_match_score = num_matches
                best_match_kf = candidate_kf
        
        if best_match_score >= self.min_matches:
            return best_match_kf
        
        return None
