import csv
import json
import numpy as np


def main():
    csv_list = []
    with open("hand_positions.csv", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            csv_list.append(row)

    with open("ledasila_handforms.json", "r") as file:
        json_list = file.read()
        json_list = json.loads(json_list)

    csv_keys = np.array(csv_list)[1:, 0].astype(dtype=int)

    json_keys = np.array(list(map(lambda x: x["value"], json_list)))

    print("Are they equal?", np.array_equal(csv_keys, json_keys))


if __name__ == "__main__":
    main()
