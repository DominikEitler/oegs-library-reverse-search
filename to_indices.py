import json
import pandas as pd

from utils import get_json_list


def main():
    json_file = get_json_list()
    df = pd.read_csv("hand_positions.csv")

    index_dict = {}
    for i, j in enumerate(json_file):
        index_dict[j["value"]] = i

    df.insert(0, "index", df["ledasila_index"].map(index_dict))
    df.to_csv("indexed_hand_pos.csv", index=False)


if __name__ == "__main__":
    main()
