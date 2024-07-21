import torch
import cv2
import time
import platform 
import sys 
import torch

sys.path.append("../../")
from config import Config
import config
config.cfg.set_lib('xfeat') 

from dataset import dataset_factory
from accelerated_features.modules.xfeat import XFeat



if __name__ == "__main__":

    config = Config()

    xfeat = XFeat()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

    dataset = dataset_factory(config.dataset_settings)

    img_id=0
    video_path = '../../videos/kitti00/video.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, previous_frame = cap.read()
    previous_frame = cv2.resize(previous_frame, (1226,370))
    while dataset.isOk():

        current_frame = dataset.getImage(img_id)
        time0=time.time()
        # ret, current_frame = cap.read()
        # if not ret:
        #     break
        mkpts_0, mkpts_1 = xfeat.match_xfeat(current_frame, previous_frame, top_k = 4096)
        
        #kp=mkpts_1.cpu().numpy()
        for k in mkpts_1:
            #print(k)
            try:
                cv2.circle(current_frame,(int(k[0]),int(k[1])),3,(255),-1)
            except:
                pass
        #print(kp)
        
        # Process matches here (e.g., for tracking, stabilization, etc.)
        
        
        #viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        # Update previous_frame for the next iteration
        img_id +=1
        previous_frame = current_frame
        cv2.imshow("frame",current_frame)
        cv2.waitKey(2)
        print(1/(time.time()-time0))
    cap.release()
