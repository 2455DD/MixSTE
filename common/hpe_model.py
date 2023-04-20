import torch
import torch.nn as nn
from scipy.spatial.distance import directed_hausdorff,euclidean
from scipy.special import softmax
import numpy as np
'''
    Refine2DNet: Posefix网络，就不实现了
    作用: 提纯2D HPE输出
    1. 避免
'''
# class Refine2DNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
        
        
'''
    人体追踪模块
    作用: 追踪人体
    NOTE: 取代了之前方法的prepare_2d_dataset()
    方法：
        - 手工方法：
            1. 第一帧选择可信度最高的
            2. 选择距离上一帧目标最近的检测框
            3. 若这一帧没有符合条件的检测框，则视为跳帧(或当作跳帧)，选择MPJPE最接近的
        - 希望增加一个面积来约束选择大小最接近的，看看行不行？面积不一定好算，但欧氏距离好算，而且能够保证和豪斯道夫距离同阶
            - 亦即目标修改为：最小化面积距离亦即
        

'''
class HumanTrackingModule():
    def calculate_matrix_difference(self,from_array:np.ndarray,to_array:np.ndarray):
        try:
            from_array=from_array.reshape(-1,1) 
            to_array=to_array.reshape(-1,1)
            if(not np.isnan(from_array).any()):            
                from_area = euclidean(from_array[0],from_array[1])
            else:
                return np.inf
            if(not np.isnan(to_array).any()):
                to_area = euclidean(to_array[0],to_array[1])
            else:
                return np.inf
            # return directed_hausdorff(from_array,to_array)[0]+abs(from_area-to_area)
            return abs(from_area-to_area)
        except IndexError as err:
            print(f'from_array:{from_array}\tShape:{from_array.shape}')
            print(f'to_array:{to_array}\tShape:{to_array.shape}')
            exit(-1)
            
    def calculate_matrix_area(self,array:np.ndarray):           
        array = array.reshape(-1,1)
        if(np.isnan(array).any()):
            return 0 
        return euclidean(array[0],array[1])
    
    def mpjpe(self,predicted, target):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        try:
            assert predicted.shape == target.shape
        except:
            predicted = predicted.T
            
        return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1))
    
    def inference(self,skeleton_arrays,bb_arrays):
        '''
            input:
                - bb_arrays:numpy array, shape: (frames,bbs)，bbs是不等长的list(ndarray)
                    - bb shape为(5),包括左上xy右下xy以及置信度分数
                - skeleton_arrays:numpy array, shape: (frames,kps)，kps是不等长的list(ndarray)
                    - kp为[17,3]
            
            output:
                - best_skeletons
                - best_bbs
        '''
        assert skeleton_arrays.shape[0] == bb_arrays.shape[0],"skeleton and bounding boxes array is not aligned"
        best_skeletons = []
        best_bbs =[]
        mpjpe_distances = []
        for i in range(skeleton_arrays.shape[0]):
            bb_arr = bb_arrays[i][1]
            if not isinstance(bb_arr,np.ndarray):
                bb_arr = np.array(bb_arr,dtype=np.float32)
                
            if len(bb_arr) == 0 or len(skeleton_arrays[i][1]) == 0:
                # No bbox/keypoints detected for this frame -> will be interpolated
                best_bb = np.full(4, np.nan, dtype=np.float32)
                best_kp = np.full((17, 4), np.nan, dtype=np.float32)
                best_bbs.append(best_bb) # 4 bounding box coordinates
                best_skeletons.append(best_kp) # 17 COCO keypoints
                continue
            if i==0:
            # 第一帧：选择置信度高的
                try:
                    best_match = np.argmax(bb_arr[:,4])
                    best_kp = skeleton_arrays[i][1][best_match].T.copy()
                    best_bb = bb_arr[best_match,:4]
                except TypeError:
                    bb_arr = np.array(bb_arrays[i][1])
                    best_match = np.argmax(bb_arr[:,4])
                    best_kp = skeleton_arrays[i][1][best_match].T.copy()
                    best_bb = bb_arrays[i][1][best_match,:4]
                except IndexError:
                    print(f"bb_arr shape:{bb_arr.shape}")
                    exit(-1)
                best_skeletons.append(best_kp)
                best_bbs.append(best_bb)
                continue
            
            # 第二帧及以后：依据上述规则
            
            differences = []
            scores = []
            for j in range(len(bb_arr)):
                current_area = self.calculate_matrix_area(bb_arr[j,:4])
                last_best_area = self.calculate_matrix_area(best_bbs[i-1])
                differences.append(self.calculate_matrix_difference(
                    bb_arr[j,:4],best_bbs[i-1]))
                scores.append(bb_arr[j,4])
                    
            differences = differences / np.linalg.norm(differences)
            scores = scores / np.linalg.norm(scores)
            
            best_diff = np.argmin(differences)
            best_scores = np.argmax(scores)
            if best_diff == best_scores:
                best_match = best_diff
            else:
                candidate_diff_skeleton = skeleton_arrays[i][1][best_diff].T.copy()
                candidate_score_skeleton = skeleton_arrays[i][1][best_scores].T.copy()
                if self.mpjpe(candidate_diff_skeleton,best_skeletons[i-1])>self.mpjpe(candidate_score_skeleton,best_skeletons[i-1]):
                    best_match = best_diff
                else:
                    best_match = best_scores
            
            best_bb = bb_arr[best_match,:4]
            best_kp = skeleton_arrays[i][1][best_match].T.copy()
            
            # 有跳帧情况下触发判别准则3
            if self.mpjpe(best_kp,best_skeletons[i-1]) >np.average(np.array(mpjpe_distances))*1.2: #如何判别跳帧
                local_mpjpes = []
                for j in range(len(skeleton_arrays[i][1])):
                    t = skeleton_arrays[i][1]
                    local_mpjpes.append(self.mpjpe(t[j],best_skeletons[i-1]))
                best_match = np.argmax(local_mpjpes)
                best_bb = bb_arr[best_match,:4]
                best_kp = skeleton_arrays[i][1][best_match].T.copy()
            
            best_skeletons.append(best_kp)
            best_bbs.append(best_bb)            
            mpjpe_distances.append(self.mpjpe(best_kp,best_skeletons[i-1]))

        best_skeletons = np.array(best_skeletons, dtype=np.float32)
        best_bbs = np.array(best_bbs, dtype=np.float32)
        return best_skeletons,best_bbs
'''
    人体遮挡判别网络
    作用：避免人体遮挡导致的估计不佳
'''
class OcclusionNetwork(nn.Module):
    def __init__():
        pass