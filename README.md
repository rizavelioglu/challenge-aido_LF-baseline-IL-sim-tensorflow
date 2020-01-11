# challenge-aido_LF-baseline-IL-sim-tensorflow
VGG16 based approach to AI Driving Olympics competition (AIDO-3)

## Steps
### 1. Donwload required packages 
`cd challenge-aido_LF-baseline-IL-sim-tensorflow/learning`
`pip install -r requirements.txt`
`pip install -e git://github.com/duckietown/gym-duckietown.git@daffy#egg=gym-duckietown`

### 2. Run "/learning/log.py" to create training data:
`python log.py`

### 3. Train VGG16 based CNN:
`python train.py`

### 4. Copy the final model from the "learning/" directory to the "submission/tf_models/" one.

### 5. Evaluate!:
`dts challenges evaluate`

### OR make an official submission:
`dts challenges submit`
