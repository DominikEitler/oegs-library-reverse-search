import cv2
import mediapipe as mp
import time
import csv
import itertools
import os

from hand_params import get_hand_parameters
from utils import draw_hand, get_json_list

mp_holistic = mp.solutions.holistic

FRAME_LIMITER = 1

DIR_PATH = "temp"


def main():
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    json_list = get_json_list()

    # if available, macbook treats iphone cam as 0, so 1 is the webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    frame_cnt = 1
    landmarks = None
    keypoints = None

    should_stop = False
    for x in json_list:
        hand_pos = x["value"]
        print(x["texts"][0], hand_pos)

        while True:
            _, image = cap.read()

            if frame_cnt % FRAME_LIMITER == 0:
                holistic_results = holistic.process(image)

                # right hand
                landmarks = holistic_results.right_hand_landmarks

                _, new_keypoints = get_hand_parameters(landmarks)

                if new_keypoints is not None:
                    keypoints = new_keypoints

                frame_cnt = 1

            image = draw_hand(image, landmarks)

            cv2.imshow("tracking...", cv2.flip(image, 1))

            frame_cnt += 1

            if cv2.waitKey(1) in [32]:
                break

            if cv2.waitKey(1) in [27]:
                print("exiting...")
                should_stop = True
                break

        if should_stop:
            break

        if keypoints is None:
            continue

        print("saving...", hand_pos, keypoints)
        keypoints_to_save = list(itertools.chain(*keypoints))

        with open(f"{DIR_PATH}/hand_positions.csv", "a", encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow([hand_pos] + keypoints_to_save)

    cv2.destroyAllWindows()
    holistic.close()
    cap.release()


if __name__ == "__main__":
    main()
