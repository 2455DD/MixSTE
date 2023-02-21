from datetime import datetime
from subprocess import check_call,CalledProcessError
import logging
from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm
import re
from common.coco_dataset import coco_h36m
from common.custom_dataset import StaticCustomDataset
from common.camera import camera_to_world,image_coordinates
from common.refine_main import refine
from common.inference_visualiz import visualize_3d_animation,prepare_2d_dataset
import numpy as np
import random

def camera_process(video_name,prediction):
    keypoints=np.load('data/data_2d_custom_' + args.dataset + '.npz',allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()
    input_keypoints = keypoints[video_name]["custom"][0].copy()
    input_keypoints,_ = coco_h36m(input_keypoints)
    dataset = StaticCustomDataset('data/data_2d_custom_' + args.dataset + '.npz')
    # Invert camera transformation
    cam = dataset.cameras()[video_name][0]
    # If the ground truth is not available, take the camera extrinsic params from a random subject.
    # They are almost the same, and anyway, we only need this for visualization purposes.
    for subject in dataset.cameras():
        if 'orientation' in dataset.cameras()[subject][0]:
            rot = dataset.cameras()[subject][0]['orientation']
            break
    prediction = camera_to_world(prediction.astype(np.float32), R=rot, t=0)
    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
    return input_keypoints,prediction


def process_single_video(keypoint):
    video_name=os.path.basename(re.findall('(.*)\.npz',keypoint)[0])
    logging.info(f"Processing {keypoint}")
    
    # ---------------- 3D 推断 ---------------------
    # specify the custom dataset (-d custom), the input keypoints as exported in the previous step (-k myvideos), 
    # the correct architecture/checkpoint, and the action custom (--viz-action custom). 
    # The subject is the file name of the input video, and the camera is always 0.
    input_video = os.path.join(args.input_video,video_name)
    if os.path.exists(input_video):
        output_keypoint=os.path.join(args.output,f"{video_name}")
        output_video=os.path.join(args.video_output,f"{video_name}_raw.mp4")
        assert not os.path.exists(output_video),f"Output Video {output_video} should be empty"
        try:
            check_call(["python","run.py",
                        "-d","custom",
                        "-k",f"{args.dataset}",
                        "-c","checkpoint",
                        "--evaluate","best_epoch.bin",
                        "--render","--viz-subject",f"{video_name}",
                        "--viz-action","custom",
                        "--viz-camera","0",
                        "--viz-output",f"{output_video}",
                        "--viz-video",f"{input_video}",
                        "--viz-export",f"{output_keypoint}",
                        "--viz-limit",f"{args.limit if args.limit>1 else 50}",
                        "--viz-downsample","2","--viz-size","3",
                        "-g",args.gpu if args.gpu is not None else "0" 
                        ])
        except Exception as e:
            print("Error occurred during execution at " + str(datetime.now().date()) + " {}".format(datetime.now().time()))
            print(e)
            exit(1)
    
    else:
        logging.warning(f"{input_video} is not found")
        output_keypoint=os.path.join(args.output,f"{video_name}")
        try:
            check_call(["python","run.py",
                        "-d","custom",
                        "-k",f"{args.dataset}",
                        "-c","checkpoint",
                        "--evaluate","best_epoch.bin",
                        "--render","--viz-subject",f"{video_name}",
                        "--viz-action","custom",
                        "--viz-camera","0",
                        "--viz-export",f"{output_keypoint}",
                        "--viz-downsample","2","--viz-size","3",
                        "-g",args.gpu if args.gpu is not None else "0" 
                        ])
        except Exception as e:
            print("Error occurred during execution at " + str(datetime.now().date()) + " {}".format(datetime.now().time()))
            print(e)
            exit(1)
            
    output_keypoint_list.append(output_keypoint)
    
    # --------------- 数据提纯 -----------------------   
    # NOTE: 研究怎么调用库来做提纯，现在的方法好像是只能运行
    output_keypoint = output_keypoint+'.npy'            
    raw_3d_keypoint = np.load(output_keypoint,allow_pickle=True)
    raw_2d_data,_ = prepare_2d_dataset(keypoint)
    raw_2d_keypoint = raw_2d_data[0]['keypoints']
    logging.debug(f"3d kp: {raw_3d_keypoint.shape}")
    refined_3d_keypoint=refine(keypoints_2d=raw_2d_keypoint,keypoints_3d=raw_3d_keypoint)
    logging.debug(f"DEBUG:type of refined_3d_keypoint:{type(refined_3d_keypoint)}")
    # logging.debug(f"DEBUG:value of refined_3d_keypoint:{refined_3d_keypoint}")
    refined_3d_keypoint=np.asarray(refined_3d_keypoint)
    
    # --------------- 可视化 --------------------------
    # NOTE: 研究如何改造当前可视化方法，
    output_video=os.path.join(args.video_output,f"{video_name}.mp4")
    
    _,refined_3d_keypoint=camera_process(video_name=video_name,prediction=refined_3d_keypoint)
    input_keypoints,raw_3d_keypoint=camera_process(video_name=video_name,prediction=raw_3d_keypoint)
    
    logging.info(f"Export comparing video to {output_video}")
    # visualize_3d_animation(raw_3d_keypoint,output_video,input_video,limit=args.limit if args.limit>0 and args.limit<raw_3d_keypoint.shape[0] else 50)    
    # visualize_3d_animation(keypoints=raw_3d_keypoint ,prediction=raw_3d_keypoint,
    #                        output_path=output_video,video_path=input_video,
    #                        limit=args.limit if args.limit>0 and args.limit<raw_3d_keypoint.shape[0] else 50,
    #                        refine=refined_3d_keypoint)    
    visualize_3d_animation(keypoints=input_keypoints ,prediction=raw_3d_keypoint,
                           output_path=output_video,video_path=input_video,
                           limit=args.limit if args.limit>0 and args.limit<raw_3d_keypoint.shape[0] else 50,
                           refine=refined_3d_keypoint)    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--input_video',type=str,metavar='PATH',help="input video's directory path")
    parser.add_argument('--keypoint',type=str,metavar='PATH',help="keypoints' directory path")
    parser.add_argument('--dataset',type=str,default='custom')
    parser.add_argument('--video_output',type=str,metavar='PATH')
    parser.add_argument('--output',type=str,metavar='PATH')
    parser.add_argument('--debug',action='store_true',default=None)
    parser.add_argument('--limit',type=int,default=-1,help='limit frame of output video')
    parser.add_argument('--sample',type=int,default=-1,help='take certain sample from input')
    parser.add_argument('--gpu',type=str)

    args = parser.parse_args()
    assert os.path.isdir(args.input_video) 
    assert os.path.isdir(args.keypoint) 
    assert args.dataset is not None
    assert os.path.isdir(args.video_output) 
    assert os.path.isdir(args.output) 
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.gpu is None:
        logging.warning("No GPU is specified, use gpu 0 instead")
    
    logging.info("Start")
    keypoint_list = glob(os.path.join(args.keypoint,'*.npz'))
    output_keypoint_list = []
    if args.sample > 0:
        logging.info(f"Sampling {args.sample} files from {args.keypoint}(length = {len(keypoint_list)})")
        logging.debug(type(keypoint_list))
        sample_list = random.sample(keypoint_list,args.sample)
        for keypoint in tqdm(sample_list):
            process_single_video(keypoint=keypoint)
    else:
        for keypoint in tqdm(keypoint_list):
            process_single_video(keypoint=keypoint)
    
