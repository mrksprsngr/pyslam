import numpy as np
import cv2
import math
import time 
from datetime import datetime


import platform 

from config import Config
# from system import Slam
#from slam_states import SlamState
from slam import Slam, SlamState

from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 

from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters  

# Constants and Configurations
NUM_FEATURES = 2000
TRACKER_TYPE = FeatureTrackerTypes.DES_BF  # Descriptor-based, brute force matching with knn
TRACKER_CONFIG = FeatureTrackerConfigs.ORB2
TRACKER_CONFIG['num_features'] = NUM_FEATURES
TRACKER_CONFIG['tracker_type'] = TRACKER_TYPE
START_IMG_ID = 2700  # Start from a desired frame id if needed
SHOW_GUI = True
GT_PATH = None   
STEP_BY_STEP = False     
PAUSED = False
VERBOSITY_LEVEL = 1
VERBOSITY_CATEGORY = {'tracking','gui','system'}
    
def configure_slam():
    config = Config()
    Printer.set_verbosity(VERBOSITY_LEVEL, VERBOSITY_CATEGORY)
    dataset = dataset_factory(config.dataset_settings)
    groundtruth = GT_PATH  # Not actually used by Slam() class; could be used for evaluating performances
    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])

    feature_tracker = feature_tracker_factory(**TRACKER_CONFIG)
    return config, dataset, groundtruth, cam, feature_tracker

def initialize_display_and_viewer(config, cam):
    if SHOW_GUI:
        viewer3D = Viewer3D()
        display2d = Display2D(cam.width, cam.height) if platform.system() == 'Linux' else None
        matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches', title='# matches')
    else:
        viewer3D = None
        display2d = None
        matched_points_plt = None

    return SHOW_GUI, viewer3D, display2d, matched_points_plt

def process_frame(slam, dataset, viewer3D, display2d, matched_points_plt, img_id):

    Printer.normal(2,6,"Dataset is ok: ", dataset.isOk())
    Printer.normal(2,6,'image: ', img_id)

    img = dataset.getImage(img_id)
    if img is None:
        Printer.normal(2,2,'image is empty')
        getchar()

    timestamp = dataset.getTimestamp()
    next_timestamp = dataset.getNextTimestamp()
    frame_duration = next_timestamp - timestamp
    Printer.normal(2,6,'frame duration: ', frame_duration)
    if img is not None:
        time_start = time.time()
        slam.track(img, img_id, timestamp)  # Main SLAM function
        update_displays(slam, viewer3D, display2d, matched_points_plt, img, img_id)
        duration = time.time() - time_start
        if frame_duration > duration:
            time.sleep(frame_duration - duration)

def update_displays(slam, viewer3D, display2d, matched_points_plt, img, img_id):
    if viewer3D is not None:
        viewer3D.draw_map(slam)
    
    img_draw = slam.map.draw_feature_trails(img)
    
    if display2d is not None:
        display2d.draw(img_draw)
    
    if matched_points_plt is not None:
        update_plot(slam, matched_points_plt, img_id)
    
def update_plot(slam, matched_points_plt, img_id):
    if slam.tracking.num_matched_kps is not None:
        matched_kps_signal = [img_id, slam.tracking.num_matched_kps]
        matched_points_plt.draw(matched_kps_signal, '# keypoint matches', color='r')
    
    if slam.tracking.num_inliers is not None:
        inliers_signal = [img_id, slam.tracking.num_inliers]
        matched_points_plt.draw(inliers_signal, '# inliers', color='g')
    
    if slam.tracking.num_matched_map_points is not None:
        valid_matched_map_points_signal = [img_id, slam.tracking.num_matched_map_points]
        matched_points_plt.draw(valid_matched_map_points_signal, '# matched map pts', color='b')
    
    if slam.tracking.num_kf_ref_tracked_points is not None:
        kf_ref_tracked_points_signal = [img_id, slam.tracking.num_kf_ref_tracked_points]
        matched_points_plt.draw(kf_ref_tracked_points_signal, '# $KF_{ref}$  tracked pts', color='c')
    
    if slam.tracking.descriptor_distance_sigma is not None:
        descriptor_sigma_signal = [img_id, slam.tracking.descriptor_distance_sigma]
        matched_points_plt.draw(descriptor_sigma_signal, 'descriptor distance $\sigma_{th}$', color='k')
    
    matched_points_plt.refresh()

def handle_gui(viewer3D, display2d, matched_points_plt, slam, do_step, is_paused):
    key = matched_points_plt.get_key() if matched_points_plt else None
    key_cv = cv2.waitKey(1) & 0xFF if display2d else None

    if slam.tracking.state == SlamState.LOST:
        if display2d:
            getchar()
        else:
            key_cv = cv2.waitKey(0) & 0xFF
    
    if do_step:
        if display2d:
            getchar()
        else:
            key_cv = cv2.waitKey(0) & 0xFF
    
    if key == 'd' or (key_cv == ord('d')):
        do_step = not do_step
        Printer.normal(2,0,'do step: ', do_step)
    
    if key == 'q' or (key_cv == ord('q')):
        if display2d:
            display2d.quit()
        if viewer3D:
            viewer3D.quit()
        if matched_points_plt:
            matched_points_plt.quit()
        exit()
    
    if viewer3D:
        is_paused = not viewer3D.is_paused()

def finish_sequence(slam, viewer3D, display2d, matched_points_plt, ts_run):
    lKfIds = slam.tracking.tracking_history.get_kf_ids()
    Printer.normal(2,0,'KF ids: ', lKfIds)
    slam.quit()
    #  slam.save_trajectory(ts_run)
    slam.save_map_keyframe_trajectory(ts_run)
    
    if viewer3D:
        viewer3D.quit()
    if display2d:
        display2d.quit()
    if matched_points_plt:
        matched_points_plt.quit()
        
def run_slam():
    config, dataset, groundtruth, cam, feature_tracker = configure_slam()
    bShowGUI, viewer3D, display2d, matched_points_plt = initialize_display_and_viewer(config, cam)

    slam = Slam(cam, feature_tracker, groundtruth)
    time.sleep(1)  # To show initial messages

    do_step = STEP_BY_STEP
    is_paused = PAUSED
    img_id = START_IMG_ID
    ts_run = datetime.now()
    Printer.normal(1,7,"Current Time =", ts_run)

    while dataset.isOk():
        if not is_paused:
            process_frame(slam, dataset, viewer3D, display2d, matched_points_plt, img_id)
            img_id += 1
        else:
            time.sleep(1)

        if bShowGUI:
            handle_gui(viewer3D, display2d, matched_points_plt, slam, do_step, is_paused)

    finish_sequence(slam, viewer3D, display2d, matched_points_plt, ts_run)

if __name__ == "__main__":
    run_slam()