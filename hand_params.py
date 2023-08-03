import numpy as np
from scipy.spatial.transform import Rotation as R

from quaternion import quaternion_from_forward_and_up_vectors
from utils import to_np, normalize, get_hand_normal_vector, get_hand_up_vector


def get_orientation_from_activations(activations):
    def pos_or_neg_value(value, positive, negative):
        if value > 0:
            return positive
        if value < 0:
            return negative

    front = pos_or_neg_value(activations[2], "front", "back")
    up = pos_or_neg_value(activations[1], "up", "down")
    side = pos_or_neg_value(activations[0], "left", "right")
    return ", ".join(filter(lambda x: bool(x), [front, up, side]))


def get_direction_activations(normalized_vector):
    THRESHOLD = 0.3

    def threshold_fn(x):
        if abs(x) <= THRESHOLD:
            return 0
        return -1 if x < 0 else 1

    return list(map(threshold_fn, normalized_vector))


def get_hand_orientation(normal_vector):
    direction_activations = get_direction_activations(normalize(normal_vector))
    orientations = get_orientation_from_activations(direction_activations)

    return orientations


def get_rotation_matrix(normal_vector, up_vector):
    q = quaternion_from_forward_and_up_vectors(normal_vector, up_vector)
    return R.from_quat(q).as_matrix()


def transform_hand_to_root(key_points, rotation_matrix):
    nparray = np.array(list(map(lambda x: to_np(x), key_points)))
    return np.dot(nparray - nparray[0], rotation_matrix)


def get_hand_keypoints(key_points, normal_vector, up_vector):
    rotation = get_rotation_matrix(normal_vector, up_vector)
    transformed_key_points = transform_hand_to_root(key_points, rotation)

    return transformed_key_points


def get_hand_parameters(hand_landmarks):
    if hand_landmarks and hand_landmarks.landmark:
        key_points = hand_landmarks.landmark

        normal_vector = get_hand_normal_vector(key_points)
        up_vector = get_hand_up_vector(key_points)

        orientation = get_hand_orientation(normal_vector)
        keypoints = get_hand_keypoints(key_points, normal_vector, up_vector)

        return orientation, keypoints
    return None, None
