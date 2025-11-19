This is a repository containing my implementation of a SLAM algorithm, utilizing stereo images from the KITTI dataset. Each timestep contains a pair of grayscale stereo images. 
  
The main structure of this repository is as follows:  
* utils.py - This file contains the following classes:
    * Keyframe - This class contains information for each keyframe that is extracted, and utilized for detecting loop-closures, optimizing the predicted trajectory, and storing the pointcloud from each keyframe.
    * BoWDatabase - This class implements a Bag of Words database, which is trained on a set of keyframes at the start of any trajectory. This allows for efficient detection of matched keyframes for loop-closure detection.
    * SpatialIndex - This class implements a KD-tree based upon the 3D positions of each keyframe. This allows for efficient querying of existing keyframes near a 3D position.
* g2o_optimizer.py - This file implements a pose-graph optimizer from the g2o library, an open-source graph optimization library. 
* loop_closure.py - This file implements a loop closure detector class, which combines a BoWDatabase and a SpatialIndex KD-tree to detect if new keyframes added to the trajectory are loop closures in the trajectory.
* slam.py - This file implements all functions necessary for the visual-odometry trajectory prediction, determining when to create a new keyframe, and building a global pointcloud.
* main.py - This file calls the functions necessary from *slam.py* to fully implement the SLAM algorithm for a given KITTI sequence, and save the visual-odometry trajectory, optimized trajectory, and global pointcloud created by this algorithm.
