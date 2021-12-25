import platform
import numpy as np
import cv2
import math
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# edge tpu delegate names for different hardware
EDGETPU_SHARED_LIB = {
    'Windows': 'edgetpu.dll',
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib'
}[platform.system()]

class Grasp:

    def __init__(self, center, angle, length, width):
        self.center = center
        self.angle = angle
        self.length = length
        self.width = width


def sqr_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    half_sm_dim = min(h, w)//2
    img = img[(h//2)-(half_sm_dim):(h//2)+(half_sm_dim), (w//2)-(half_sm_dim):(w//2)+(half_sm_dim)]
    return img

def detect_grasps(p_img, width_img, ang_img, num_grasps=1, ang_threshold=5):
    # TODO: check speed
    local_max = peak_local_max(p_img, min_distance=20, threshold_abs=0.2, num_peaks=num_grasps)

    grasps = []

    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_length = width_img[grasp_point]
        g_width = grasp_length/2
        grasp_angle = ang_img[grasp_point]

        if ang_threshold > 0:
            if grasp_angle > 0:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].max()
            else:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].min()


        g = Grasp(grasp_point, grasp_angle, grasp_length, g_width)

        grasps.append(g)

    return grasps


def main():
    # file directories
    model_dir = 'trained_models/ggcnn_model_edgetpu.tflite'
    img_dir = 'sample_imgs/img_1.jpeg'

    # set interpreter and get tensors
    interpreter = Interpreter(model_dir, experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)])
    interpreter.allocate_tensors()

    # get input/output tensor information
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    input_scale, input_zero_point = interpreter.get_input_details()[0]['quantization']
    data_type = interpreter.get_input_details()[0]['dtype']

    output_scale_0, output_zero_point_0 = interpreter.get_output_details()[0]['quantization']
    output_scale_1, output_zero_point_1 = interpreter.get_output_details()[1]['quantization']
    output_scale_2, output_zero_point_2 = interpreter.get_output_details()[2]['quantization']
    output_scale_3, output_zero_point_3 = interpreter.get_output_details()[3]['quantization']

    # read and manipulate image
    img = cv2.imread(img_dir)
    img_sqr = sqr_crop(img)
    img_resized = cv2.resize(img_sqr, (input_height, input_width))
    frame = img_resized
    img_scaled = (img_resized / input_scale) + input_zero_point # scale input for quantized model
    img_expanded = np.expand_dims(img_scaled, axis=0).astype(data_type)

    # run model
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_expanded)
    interpreter.invoke()

    # get model output
    model_output_data_0 = (interpreter.get_tensor(interpreter.get_output_details()[0]['index'])).astype(np.float32)
    model_output_data_1 = (interpreter.get_tensor(interpreter.get_output_details()[1]['index'])).astype(np.float32)
    model_output_data_2 = (interpreter.get_tensor(interpreter.get_output_details()[2]['index'])).astype(np.float32)
    model_output_data_3 = (interpreter.get_tensor(interpreter.get_output_details()[3]['index'])).astype(np.float32)

    # scale output from quantized model
    model_output_data_0 =(model_output_data_0 - output_zero_point_0) * output_scale_0
    model_output_data_1 =(model_output_data_1 - output_zero_point_1) * output_scale_1
    model_output_data_2 =(model_output_data_2 - output_zero_point_2) * output_scale_2
    model_output_data_3 =(model_output_data_3 - output_zero_point_3) * output_scale_3

    # get postion, angle, and width maps
    grasp_positions_out = model_output_data_0
    grasp_angles_out = np.arctan2(model_output_data_2, model_output_data_1)/2.0
    grasp_width_out = model_output_data_3 * 150

    # convert maps to images
    grasp_position_img = grasp_positions_out[0, ].squeeze()
    grasp_width_img = grasp_width_out[0, ].squeeze()
    grasp_angles_img = grasp_angles_out[0, ].squeeze()

    # run each image through a gaussian filter
    grasp_position_img = gaussian(grasp_position_img, 2.0, preserve_range=True)
    grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)
#    grasp_angles_img = gaussian(grasp_angles_img, 2.0, preserve_range=True)

    # find grasps from images
    grasps = detect_grasps(grasp_position_img, grasp_width_img, grasp_angles_img)

    for g in grasps:
        line_x1 = int(g.center[1]-(g.width*math.cos(g.angle)))
        line_y1 = int(g.center[0]+(g.width*math.sin(g.angle)))
        line_x2 = int(g.center[1]+(g.width*math.cos(g.angle)))
        line_y2 = int(g.center[0]-(g.width*math.sin(g.angle)))

        cv2.line(frame, (line_x1,line_y1), (line_x2,line_y2), (0, 0, 255), 2)

    frame = cv2.resize(frame, (600, 600))
    cv2.imshow("Frame", frame)
    cv2.waitKey(1000)

if __name__ == '__main__':
    main()
