import argparse
from glob import glob
import logging
import random
import numpy as np
import os
import re
# from common.visualization import render_animation,read_video,get_fps,get_resolution
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,writers,ArtistAnimation
from matplotlib.axes import Axes
from detectron2.utils.visualizer import Visualizer,VisImage
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
import subprocess as sp
from torch import from_numpy
from tqdm import tqdm

from common.visualization import render_animation 
from common.h36m_dataset import  StaticHuman36mDataset,h36m_metadata
from common.camera import camera_to_world, image_coordinates
from common.custom_dataset import StaticCustomDataset
import copy
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from common.coco_dataset import coco_h36m
from common.hpe_model import HumanTrackingModule
#------------------ Macro-----------------
_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05

human36m_skeleton = StaticHuman36mDataset().skeleton()
#----------------------------------------

def camera_process(video_name,prediction):
    keypoints=np.load('data/data_2d_custom_' + "MitSkating" + '.npz',allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()
    input_keypoints = keypoints[video_name]["custom"][0].copy()
    input_keypoints,_ = coco_h36m(input_keypoints)
    dataset = StaticCustomDataset('data/data_2d_custom_' + "MitSkating" + '.npz')
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


def get_parser():
    parser = argparse.ArgumentParser(description="2D/3D Keypoint Visualization Tool")
    # Argument 
    parser.add_argument('-t','--type',type=str,default='',help="Visualization type.2D or 3D")
    parser.add_argument('-i','--input',type=str,metavar='PATH',help="keypoint directory")
    parser.add_argument('-v','--video',type=str,metavar='PATH',help="video directory, leave it blank to disable this function")
    parser.add_argument('-o','--output',type=str,metavar='PATH',help="output file name, endding with .gif or .mp4")
    parser.add_argument('-f','--force',action='store_true',help='override the output file')
    parser.add_argument('-a','--cfg',type=str,default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
                        help="configuration file name in detectron2/modelzoo")
    parser.add_argument('--refine',type=str,metavar='PATH',default=None,help='refine input')
    parser.add_argument('--output_path',type=str,metavar='PATH',default=None,help='video path for automatic input')
    parser.add_argument('--limit',type=int,default=-1,help='limit of viz frame')
    parser.add_argument('--sample',type=int,default=-1,metavar='N',help="choose sample")
    args = parser.parse_args()
    
    if not args.type:
        logger.error("Please specify the input type")
        exit(0)
    elif not args.type in ['2D','3D']:
        logger.error("Type should be '2D' or '3D'")
        exit(0)
    
    if not args.input:
        logger.error("Please specify the input directory")
        exit(0)
    elif not os.path.exists(args.input) and not os.path.isdir(args.input):
        logger.error("input file not exists")
        exit(0)
    
    if not args.video:
        logger.warning("No Video is specify, working in no-video output mode")
        
    if not args.output:
        logger.error("Please specify the output directory")
        exit(0)
    elif os.path.basename(args.output).split(".")[1] not in ["gif","mp4"]:
        logger.error("illegal output format(only mp4 and gif),stop")
        exit(0)
    elif os.path.exists(args.output) and not args.force:
        logger.error("output path already exists. delete it or use --force/-f,stop")
        exit(0)
    return parser
 
def setup_cfg(args):
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    cfg.freeze()
    return cfg

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)

def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    # w = 1000
    # h = 1002

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-hide_banner',
               '-loglevel','warning',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))




# NOTE: inference/prepare_data_2d_custom.py::decode()
# deal with single file
# Interpolated if data is missing
# NOTE: metadata是一个字典，包括w和h
def prepare_2d_dataset(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    decoder = HumanTrackingModule()
    print('Processing {}'.format(filename))
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata'].item()
    
    kp,bb = decoder.inference(kp,bb)

    # print(f"Skeletons Shape:{kp.shape}\tSkeleton[0] Shape: {kp[0].shape}")
    kp = kp[:, :, :2] # Extract (x, y)
    
    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])
    
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    print('----------')
    
    return [{
        'start_frame': 0, # Inclusive
        'end_frame': len(kp), # Exclusive
        'bounding_boxes': bb,
        'keypoints': kp,
    }], metadata       
     
def draw_and_connect_keypoints(ax:Axes,keypoints,width,height):
    """
    Draws keypoints of an instance and follows the rules for keypoint connections
    to draw lines between appropriate keypoints. This follows color heuristics for
    line color.

    Args:
        keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
            and the last dimension corresponds to (x, y, probability).

    Returns:
        output (VisImage): image object with visualizations.
    """
    # 组装VisImage
    img = np.asarray(np.empty([height,width,3]).clip(0, 255).astype(np.uint8))
    output = VisImage(img, scale=1.0)
    
    _default_font_size=max(
            np.sqrt(height * width) // 90, 10 // 1.0
        )
    
    def draw_line(x_data, y_data, color, linestyle="-", linewidth=None):
        nonlocal ax
        if linewidth is None:
            linewidth = _default_font_size / 3
        linewidth = max(linewidth, 1)
        output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * 1.0,
                color=color,
                linestyle=linestyle,
            )
        )
      
    output.ax.set_facecolor('white')
    visible = {}
    keypoint_names = MetadataCatalog.get("keypoints_coco_2017_val").get("keypoint_names")
    for idx, keypoint in enumerate(keypoints):

        # draw keypoint
        x, y= keypoint
        output.ax.add_patch(mpl.patches.Circle((x,y), radius=3, fill=True, color=_RED))
        if keypoint_names:
            keypoint_name = keypoint_names[idx]
            visible[keypoint_name] = (x, y)

    if MetadataCatalog.get("keypoints_coco_2017_val").get("keypoint_connection_rules"):
        for kp0, kp1, color in MetadataCatalog.get("keypoints_coco_2017_val").keypoint_connection_rules:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                color = tuple(x / 255.0 for x in color)
                draw_line([x0, x1], [y0, y1], color=color)

    # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
    # Note that this strategy is specific to person keypoints.
    # For other keypoints, it should just do nothing
    try:
        ls_x, ls_y = visible["left_shoulder"]
        rs_x, rs_y = visible["right_shoulder"]
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    except KeyError:
        pass
    else:
        # draw line from nose to mid-shoulder
        nose_x, nose_y = visible.get("nose", (None, None))
        if nose_x is not None:
            draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=_RED)

        try:
            # draw line from mid-shoulder to mid-hip
            lh_x, lh_y = visible["left_hip"]
            rh_x, rh_y = visible["right_hip"]
        except KeyError:
            pass
        else:
            mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
            draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=_RED)
            
    return output

# 借鉴 detectron2方法
def visualize_2d_animation(metadata,keypoints,video_path,size,output_path,fps,limit=-1):
    # Create Plot
    
    # # 注册数据集
    # DatasetCatalog.register("custom",prepare_2d_dataset)
    
    # # 组装metadata
    # MetadataCatalog.get("custom").set()

    # 创建图
    fig = plt.figure(figsize=(2*size,size))
    subplt_video = fig.add_subplot(1,2,1)
    subplt_output = fig.add_subplot(1,2,2)
    subplt_output.set_title("2D Prediction")
    subplt_video.set_title("Input Video")
    
    # 读取视频 
    if video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], metadata['h'], metadata['w']), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(video_path,limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        if fps is None:
            fps = get_fps(video_path)
 
    
    # if metadata is not None:
    #     v = Visualizer(img_rgb=np.empty([metadata['h'],metadata['w'],3]),metadata=MetadataCatalog.get("keypoints_coco_2017_val"))
    # else:
    #     v = Visualizer(img_rgb=np.empty([metadata['h'],metadata['w'],3]),metadata=MetadataCatalog.get("keypoints_coco_2017_val"))
    # Type: VisImage
    image_sequence = []
    
    if limit>1:
        effective_length = min(limit,effective_length)
    # keypoints = 
    # for frame_idx in range(effective_length):
    #     frame = keypoints[frame_idx][:][:]
    #     append_row = np.ones((frame.shape[0],1),dtype=np.float64)
    #     frame=np.append(frame,append_row,axis=1)
    #     single_image = v.draw_and_connect_keypoints(from_numpy(frame))
    #     if single_image is not None:
    #         # Convert to image
    #         # subplt_video.imshow(all_frames[frame_idx],animated=True)
    #         # subplt_output.imshow(single_image.img,animated=True)
    #         if frame_idx == 10:
    #             single_image.save("test.jpg")
    #         image_sequence.append(single_image.get_image())

    initialized = False
    video_image = None    
    output_image = None
    
    #--------------------------------
    #   FuncAnimation 方法
    #   现在的问题是会重影
    #
    #
    #--------------------------------
    # output_image=subplt_output.imshow(all_frames[0])
    # video_image=subplt_video.imshow(image_sequence[0])
    
    
    def update_video(i):
        if i%10==0:
            logger.debug(f"Update Animation: {i}")
        nonlocal initialized,video_image,output_image,subplt_output
        if not initialized:
            output_visimage=draw_and_connect_keypoints(subplt_output,keypoints=keypoints[i][:][:],width=metadata['w'],height=metadata['h'])
            output_image=subplt_output.imshow(output_visimage.get_image())
            video_image=subplt_video.imshow(all_frames[i])
            initialized=True
        else:
            video_image.set_data(all_frames[i])
            output_visimage=draw_and_connect_keypoints(subplt_output,keypoints=keypoints[i][:][:],width=metadata['w'],height=metadata['h'])
            subplt_output.set_title("2D Prediction")
            output_image=subplt_output.imshow(output_visimage.get_image())
            
        return output_image,video_image
    
    def init():
        subplt_output.set_facecolor('white')
        logger.info("animation init")
        subplt_output.clear()
        subplt_video.clear()
        # subplt_output.set_title("2D Prediction")
        # subplt_video.set_title("Input Video")
        output_image=subplt_output.imshow(np.zeros(all_frames[0].shape))
        video_image=subplt_video.imshow(np.zeros(all_frames[0].shape))
        return output_image,video_image
            
    anim = FuncAnimation(fig, update_video,init_func=init, frames=np.arange(0, effective_length), interval=1000/fps, repeat=False,blit=True)
    
    
    #-----------------------------
    #   AritstAnimation 方法
    #
    #
    #-----------------------------
    # input_video_frame_sequence = []
    # output_video_frame_sequence = []
    # ims =[]
    
    # # 生成Artists
    # for i in range(effective_length):
    #     if initialized:
    #         # output_image.set_data(np.zeros(image_sequence[0].shape))
    #         video_image = subplt_video.imshow(all_frames[i],animated=True) 
    #         output_image=subplt_output.imshow(image_sequence[i],animated=True)
    #     if not initialized:
    #         video_image=subplt_video.imshow(all_frames[i])
    #         output_image=subplt_output.imshow(image_sequence[i])
    #         initialized=True
    #     ims.append([video_image,output_image])    

    # anim = ArtistAnimation(fig, ims,interval=1000/fps,repeat=False, blit=True)
    
    if output_path.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=3000)
        logger.info("Start outputing video")
        anim.save(output_path, writer=writer)
    elif output_path.endswith('.gif'):
        logger.info("Start outputing gif")
        anim.save(output_path, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()   

def visualize_3d_animation(prediction,output_path,video_path,keypoints=None,limit=60,refine=None):
    try:
        logger
    except NameError as e:
        logging.warning("Logger in inference_visualiz.py is not defined, use default instead")
        logger = logging
        logger.basicConfig(level=logging.INFO)
        
    logger.info('Rendering...')
    
    if keypoints is None:
        keypoints=prediction
        
    if refine is not None:
        anim_output={'Refined Output':refine}
        anim_output['Raw Output']=prediction
    else:
        anim_output={"3D Output":prediction}
    
    render_animation(
        keypoints=keypoints,
        keypoints_metadata=h36m_metadata,
        poses=anim_output,
        skeleton=copy.deepcopy(human36m_skeleton),
        output=output_path,
        azim=70, # 也可以试试±110和-70
        fps=get_fps(video_path) if video_path is not None else 25,
        viewport=(prediction.shape[1],prediction.shape[2]),
        bitrate=3000,
        # downsample=2,
        # size=3,
        input_video_path=video_path,
        limit=limit
    )

if __name__ == '__main__':
    logger=setup_logger()
    logger.info("start")
    
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    # 2D or 3D
    if args.type=='2D':
        if os.path.isdir(args.input):
            assert(os.path.isdir(args.output_path)) # WARNING: output_path是指视频路径
            file_list=glob(args.input+'/*.npz')
            if args.sample != -1:
                file_list = random.choices(file_list,k=args.sample)
            for f in tqdm(file_list):
                logger.info(f"visualizing {f}")
                basename = os.path.basename(f)
                video_basename = os.path.splitext(basename)[0]
                video_path = os.path.join(args.video,video_basename)
                output_path = os.path.join(args.output_path,video_basename+'_2d.mp4')
                if not os.path.isfile(video_path):
                    logger.warning(f"{video_path} not exists,skipping")
                    continue
                data,video_metadata=prepare_2d_dataset(f)
                video_metadata['w'],video_metadata['h'] = get_resolution(video_path)
                visualize_2d_animation(metadata=video_metadata,keypoints=data[0]['keypoints'],video_path=video_path,size=3,output_path=output_path,fps=None,
                                       limit=args.limit)
        else:
            data,video_metadata=prepare_2d_dataset(args.input)
            # logger.debug(f'{data[0].keys()}')
            video_metadata['w'],video_metadata['h'] = get_resolution(args.video)
            visualize_2d_animation(metadata=video_metadata,keypoints=data[0]['keypoints'],video_path=args.video,size=3,output_path=args.output,fps=None,
                                   limit=args.limit)
    else:
        if os.path.isdir(args.input):
            assert(os.path.isdir(args.output_path))
            file_list=glob(args.input+f'/*.npy')
            if args.sample != -1:
                    file_list = random.choices(file_list,k=args.sample)
            for f in tqdm(file_list):
                logger.info(f"visualizing {f}")
                basename = os.path.basename(f)
                
                video_basename = os.path.splitext(basename)[0]
                # video_basename = re.findall('(.*)\.npz',basename)[0]
                output_path = os.path.join(args.output_path,video_basename+'.mp4')
                video_path = os.path.join(args.video,video_basename)
                
                if os.path.splitext(f)[1]==".npz":
                    keypoints_npz=np.load(f,allow_pickle=True)
                    keypoints = keypoints_npz['raw_3d']
                elif os.path.splitext(f)[1]==".npy":
                    keypoints = np.load(f,allow_pickle=True)
                if args.refine is not None:
                    refine_npz=np.load(os.path.join(args.refine,f),allow_pickle=True)
                    refine = refine_npz['reconstruction']
                else:
                    refine = None
                _,keypoints = camera_process(video_basename,keypoints)
                visualize_3d_animation(prediction=keypoints,video_path=video_path,
                               output_path=output_path,limit=args.limit,refine=refine)
        else:
            basename = os.path.basename(args.input)
            video_basename = os.path.splitext(basename)[0]
            if os.path.splitext(args.input)[1]==".npz":
                keypoints_npz=np.load(args.input,allow_pickle=True)
                keypoints = keypoints_npz['raw_3d']
            elif os.path.splitext(args.input)[1]==".npy":
                keypoints = np.load(args.input,allow_pickle=True)
            if args.refine is not None:
                refine_npz=np.load(args.refine,allow_pickle=True)
                refine = refine_npz['reconstruction']
            else:
                refine = None
            keypoints,_ = camera_process(video_basename,keypoints)
   
            visualize_3d_animation(prediction=keypoints,video_path=args.video,output_path=args.output,limit=args.limit,refine=refine)