import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import get_json_list


def main():
    json_file = get_json_list()

    csv_file = pd.read_csv("indexed_hand_pos.csv")

    x = csv_file.drop(["index", "ledasila_index"], axis=1)
    y = csv_file["index"]

    n_inputs = csv_file.shape[1] - 2
    n_hidden = 10
    n_outputs = len(json_file)

    batch_size = 5
    learning_rate = 1.0

    data_x = torch.tensor(x.values, dtype=torch.float32)
    data_y = torch.tensor(y.values, dtype=torch.int64)

    model = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_outputs),
        nn.Softmax(dim=1),
    )

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1000):
        for i in range(0, data_x.shape[0], batch_size):
            batch_x = data_x[i : i + batch_size]
            batch_y = data_y[i : i + batch_size]

            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(epoch, loss.item())

    torch.save(model, "model.pth")


if __name__ == "__main__":
    main()
