import cv2
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

# Define key point indices based on the PoseNet model output
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

def load_model(model_path):
    """Load TFLite model, returns an Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def process_image(interpreter, image, input_index):
    """Process an image, Return a list of positions in a 4-Tuple (pos_x, pos_y, offset_x, offset_y)"""
    input_data = np.expand_dims(image, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    offset_data = np.squeeze(interpreter.get_tensor(output_details[1]['index']))

    points = []
    total_row, total_col, total_points = output_data.shape

    for k in range(total_points):
        max_score = output_data[0][0][k]
        max_row = 0
        max_col = 0
        for row in range(total_row):
            for col in range(total_col):
                if output_data[row][col][k] > max_score:
                    max_score = output_data[row][col][k]
                    max_row = row
                    max_col = col

        points.append((max_row, max_col))

    positions = []
    for idx, point in enumerate(points):
        pos_y, pos_x = point
        offset_x = offset_data[pos_y][pos_x][idx + 17]
        offset_y = offset_data[pos_y][pos_x][idx]
        positions.append((pos_x, pos_y, offset_x, offset_y))

    return positions

def display_result(positions, frame):
    """Display Detected Points in circles"""
    size = 5
    color = (255, 0, 0)
    thickness = 3

    width = frame.shape[1]
    height = frame.shape[0]

    for pos in positions:
        pos_x, pos_y, offset_x, offset_y = pos
        x = int(pos_x / 8 * width + offset_x)
        y = int(pos_y / 8 * height + offset_y)
        cv2.circle(frame, (x, y), size, color, thickness)

    cv2.imshow('Pose Detection', frame)

def is_bad_posture(positions):
    """Analyze positions and return True if bad posture is detected, otherwise False"""
    left_shoulder = positions[LEFT_SHOULDER]
    right_shoulder = positions[RIGHT_SHOULDER]
    left_hip = positions[LEFT_HIP]
    right_hip = positions[RIGHT_HIP]
    nose = positions[NOSE]

    # Calculate the average shoulder and hip height
    shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_avg_y = (left_hip[1] + right_hip[1]) / 2

    # Calculate the vertical alignment of shoulders
    shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])

    # Calculate the vertical alignment of the back
    back_alignment = abs(shoulder_avg_y - hip_avg_y)

    # Check if the head is too forward
    head_forward = abs(nose[0] - ((left_shoulder[0] + right_shoulder[0]) / 2))

    # Define thresholds for bad posture detection
    shoulder_alignment_threshold = 20  # Example value, adjust based on calibration
    back_alignment_threshold = 30  # Example value, adjust based on calibration
    head_forward_threshold = 50  # Example value, adjust based on calibration

    if shoulder_alignment > shoulder_alignment_threshold or back_alignment > back_alignment_threshold or head_forward > head_forward_threshold:
        return True
    return False

if __name__ == "__main__":
    model_path = 'data/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
    image_path = 'data/bad_posture.jpg'

    interpreter = load_model(model_path)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    input_index = input_details[0]['index']

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(frame.shape)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height))

    positions = process_image(interpreter, image, input_index)

    # Check for bad posture
    if is_bad_posture(positions):
        print("Bad posture detected!")
        # Add visual feedback on the image
        cv2.putText(frame, "Bad Posture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    display_result(positions, frame)

    key = cv2.waitKey(0)
    if key == 27:  # esc
        cv2.destroyAllWindows()
