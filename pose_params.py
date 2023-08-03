import numpy as np
from scipy.spatial.transform import Rotation as R

from quaternion import quaternion_from_forward_and_up_vectors
from utils import to_np, normalize, get_hand_normal_vector, get_hand_up_vector


def get_position_parameters(pose_landmarks):
    if pose_landmarks and pose_landmarks.landmark:
        key_points = pose_landmarks.landmark

        # print(key_points)

    return None, None
