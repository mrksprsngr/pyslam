import numpy as np
import time
from enum import Enum

from collections import defaultdict, Counter
from itertools import chain
from datetime import datetime

import cv2
import g2o

from parameters import Parameters  

from frame import Frame, match_frames
from keyframe import KeyFrame
from map_point import MapPoint
from map import Map

from search_points import propagate_map_point_matches
from search_points import search_map_by_projection, search_frame_by_projection

from local_mapping import LocalMapping
from initializer import Initializer
import optimizer_g2o

from timer import TimerFps

from slam_dynamic_config import SLAMDynamicConfig
from motion_model import MotionModel, MotionModelDamping

from feature_tracker import FeatureTrackerTypes 

from utils_sys import Printer, getchar, Logging
from utils_draw import draw_feature_matches
from utils_geom import triangulate_points, poseRt, normalize_vector, inv_T, triangulate_normalized_points, estimate_pose_ess_mat


kVerbose = True     
kTimerVerbose = False 
kDebugDrawMatches = False 

kLocalMappingOnSeparateThread = Parameters.kLocalMappingOnSeparateThread 
kTrackingWaitForLocalMappingToGetIdle = Parameters.kTrackingWaitForLocalMappingToGetIdle
kTrackingWaitForLocalMappingSleepTime = Parameters.kTrackingWaitForLocalMappingSleepTime

kLogKFinfoToFile = True 

kUseDynamicDesDistanceTh = True  

kRansacThresholdNormalized = 0.0003  # 0.0003 # metric threshold used for normalized image coordinates 
kRansacProb = 0.999
kNumMinInliersEssentialMat = 8

kUseGroundTruthScale = False 

kNumMinInliersPoseOptimizationTrackFrame = 10
kNumMinInliersPoseOptimizationTrackLocalMap = 20

kUseMotionModel = Parameters.kUseMotionModel or Parameters.kUseSearchFrameByProjection
kUseSearchFrameByProjection = Parameters.kUseSearchFrameByProjection and not Parameters.kUseEssentialMatrixFitting         
kUseEssentialMatrixFitting = Parameters.kUseEssentialMatrixFitting      
       
kNumMinObsForKeyFrameDefault = 3


class TrackingHistory(object):
    def __init__(self):
        self.relative_frame_poses = []  # list of relative frame poses as g2o.Isometry3d() (see camera_pose.py)
        self.relative_frame_kfid = []   # list of keyframe IDs corresponding to the relative frame poses
        self.kf_references = []         # list of reference keyframes  
        self.timestamps = []            # list of frame timestamps 
        self.slam_states = []           # list of slam states 

    def save_keyframe_trajectory_tum(self, filename):
        Printer.normal(2,0,f"\nSaving keyframe trajectory to {filename} ...")

        keyframes = self.kf_references
        timestamps = self.timestamps
        slam_states = self.slam_states

        # Make sure keyframes and timestamps are sorted by the keyframe ID
        keyframes, timestamps, slam_states = zip(*sorted(zip(keyframes, timestamps, slam_states), key=lambda x: x[0].id))
        with open(filename, 'w') as f:
            for i, kf in enumerate(keyframes):
                if slam_states[i]:
                    pass
                Printer.normal(2,0,"slam_states[i]: ", slam_states[i])   
                q_w = kf.Twc_as_g2oIsometry3d.orientation().w()
                q_x = kf.Twc_as_g2oIsometry3d.orientation().x()
                q_y = kf.Twc_as_g2oIsometry3d.orientation().y()
                q_z = kf.Twc_as_g2oIsometry3d.orientation().z()

                t_x, t_y, t_z = kf.Twc_as_g2oIsometry3d.position()
                ts_nano = timestamps[i]* 1e9
                f.write(f"{ts_nano:.6f} {t_x:.7f} {t_y:.7f} {t_z:.7f} {q_w:.7f} {q_x:.7f} {q_y:.7f} {q_z:.7f}\n")