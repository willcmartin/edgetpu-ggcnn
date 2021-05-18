import platform
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# edpre tpu delegate names for different hardware
EDGETPU_SHARED_LIB = {
    'Windows': 'edgetpu.dll',
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib'
}[platform.system()]

def main():
    # file directories
    model_dir = 'trained_models/ggcnn_model_edgetpu.tflite'
    img_dir = 'sample_imgs/img_1.jpeg'

    # set interpreter and get tensors
    interpreter = Interpreter(model_dir, experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)])

    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    input_scale, input_zero_point = interpreter.get_input_details()[0]['quantization']
    data_type = interpreter.get_input_details()[0]['dtype']

    img = cv2.imread(img_dir)
    img_resized = cv2.resize(img, (input_height, input_width))
    img_scaled = (img_resized / input_scale) + input_zero_point
    img_expanded = np.expand_dims(img_scaled, axis=0).astype(data_type)

    cv2.imshow("image", img_resized)
    cv2.waitKey(1000)

if __name__ == '__main__':
    main()
