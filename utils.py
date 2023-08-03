import math
import numpy as np
import mediapipe as mp
import csv
import json

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_normal_vector(p1, p2, p3):
    return np.cross(p2 - p1, p3 - p1)


def normalize(vector):
    return vector / np.linalg.norm(vector)


def to_np(point):
    return np.array([point.x, point.y, point.z])


def get_hand_normal_vector(key_points):
    p0, p1, p2 = to_np(key_points[0]), to_np(key_points[5]), to_np(key_points[17])

    return get_normal_vector(p0 - p0, p1 - p0, p2 - p0)


def get_hand_up_vector(key_points):
    p0, p1 = to_np(key_points[0]), to_np(key_points[9])
    return p1 - p0


def draw_hand(image, landmarks):
    image_copy = image.copy()
    if not landmarks:
        return image_copy

    mp_drawing.draw_landmarks(image_copy, landmarks, mp_hands.HAND_CONNECTIONS)
    return image_copy


def draw_pose(image, landmarks):
    if not landmarks:
        return image

    mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)
    return image


def get_csv_list(special=""):
    csv_list = []
    with open(f"indexed_hand_pos{special}.csv", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            csv_list.append(row)
    return np.array(csv_list[1:], dtype=float)


def get_json_list():
    json_list = []
    with open("ledasila_handforms.json", "r") as file:
        json_list = file.read()
        json_list = json.loads(json_list)
    return json_list
