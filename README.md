# Bad-Posture-Detection

## Description
This project uses TensorFlow Lite's PoseNet model to detect human poses in images and identify bad posture. The script processes an image, detects key points of the body, and checks for indicators of bad posture such as slouched shoulders, misaligned hips, and forward head position.

## Installation
To run this project, you need to have Python installed on your machine. You also need to install the required Python packages. Follow the steps below to set up your environment.

1. pip install opencv-python
2. pip install numpy
3. pip install Pillow
4. pip install tensorflow

* import tensorflow.lite as tflite (import statement)
This is the poseNet model file --> posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
Place different images in data folder along with this model.

To run it , go to terminal and enter the following command
--> python bad_posture_detection.py

