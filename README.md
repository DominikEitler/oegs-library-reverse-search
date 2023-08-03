# ÖGS library reverse search
A student project aiming to search an ÖGS (Austrian sign language) library with a sign video input.

This project uses [Google MediaPipe](https://developers.google.com/mediapipe) to detect hand key points, transforms them and then uses either k-NN or a neural network to find out the "hand shape" which can then be used to search for the (German) translation in the [LedaSila](https://ledasila.aau.at/) library

### Install dependencies (pip)
- make sure, your python version >= 3.8
- [optional] create virtual environment
- run `pip install -r requirements.txt`

### Create training/reference data
- run `python generate_hands.py`
- with the _space bar_, capture the key points of every hand shape (see command line prompt)
  - use the [hand shape reference on the LedaSila search](https://ledasila.aau.at/Search/SearchExt.aspx) if the hand shape description is not clear
- afterwards, run `python to_indices.py`

### Train the neural network [optional]
- run `python train_model.py`

### Perform the hand shape classification
- run `python detect.py`
