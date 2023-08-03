NN = False
KNN = not NN

import numpy as np
import cv2
import mediapipe as mp

from hand_params import get_hand_parameters
from pose_params import get_position_parameters
from utils import draw_hand, draw_pose, get_csv_list, get_json_list

if KNN:
    from sklearn.neighbors import KNeighborsClassifier
if NN:
    import torch


mp_holistic = mp.solutions.holistic

FRAME_LIMITER = 4
N_NEIGHBORS = 3


def main():
    csv_list = get_csv_list()
    json_list = get_json_list()

    def get_label(json_index):
        return next(filter(lambda x: x["value"] == json_index, json_list))["texts"][0]

    def get_results(dist, ind):
        csv_indices = csv_list[ind[0]][:, 1].astype(int)
        labels = [get_label(index) for index in csv_indices]
        return labels

    if NN:
        model = torch.load("model.pth")
        model.eval()

    if KNN:
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
        knn.fit(csv_list[:, 2:], csv_list[:, 1])

    should_stop = False

    # if available, macbook treats iphone cam as 0, so 1 is the webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    frame_cnt = 1

    right_hand_landmarks = None
    right_hand_keypoints = None

    # left_hand_landmarks = None
    # left_hand_keypoints = None

    # pose_landmarks = None
    # pose_keypoints = None

    while not should_stop:
        _, image = cap.read()

        if frame_cnt % FRAME_LIMITER == 0:
            # get landmarks
            holistic_results = holistic.process(image)
            right_hand_landmarks = holistic_results.right_hand_landmarks
            # left_hand_landmarks = holistic_results.left_hand_landmarks
            # pose_landmarks = holistic_results.pose_landmarks

            # right hand
            _, new_right_hand_keypoints = get_hand_parameters(right_hand_landmarks)
            if new_right_hand_keypoints is not None:
                right_hand_keypoints = new_right_hand_keypoints

            # # left hand
            # _, new_left_hand_keypoints = get_hand_parameters(left_hand_landmarks)
            # if new_left_hand_keypoints is not None:
            #     left_hand_keypoints = new_left_hand_keypoints

            # # pose
            # _, new_pose_keypoints = get_position_parameters(pose_landmarks)
            # if new_pose_keypoints is not None:
            #     pose_keypoints = new_pose_keypoints

            frame_cnt = 1

        # draw landmarks
        image = draw_hand(image, right_hand_landmarks)
        # image = draw_hand(image, left_hand_landmarks)
        # image = draw_pose(image, pose_landmarks)

        cv2.imshow("tracking...", cv2.flip(image, 1))

        if right_hand_keypoints is not None:
            if KNN:
                # knn multiple
                dist, ind = knn.kneighbors([right_hand_keypoints.flatten()])
                results = get_results(dist, ind)
                print(f"knn:", results)

            if NN:
                # model
                tensor = torch.tensor(right_hand_keypoints.flatten()).float()
                pred = model(torch.unsqueeze(tensor, 0))
                pred = torch.argmax(pred).item()
                index = int(csv_list[pred, 1])
                print("model:", get_label(index))

        frame_cnt += 1

        if cv2.waitKey(1) in [27, 32]:
            should_stop = True

    cv2.destroyAllWindows()
    holistic.close()
    cap.release()


if __name__ == "__main__":
    main()
