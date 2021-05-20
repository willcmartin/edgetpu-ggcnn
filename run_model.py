import platform
import numpy as np
import cv2
import math
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# edpre tpu delegate names for different hardware
EDGETPU_SHARED_LIB = {
    'Windows': 'edgetpu.dll',
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib'
}[platform.system()]

def sqr_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    half_sm_dim = min(h, w)//2
    img = img[(h//2)-(half_sm_dim):(h//2)+(half_sm_dim), (w//2)-(half_sm_dim):(w//2)+(half_sm_dim)]
    return img

def main():
    # file directories
    model_dir = 'trained_models/ggcnn_model_edgetpu.tflite'
#    img_dir = 'sample_imgs/img_1.jpeg'
    img_dir = 'sample_imgs/test_4.png'

    # set interpreter and get tensors
    interpreter = Interpreter(model_dir, experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB)])

    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    input_scale, input_zero_point = interpreter.get_input_details()[0]['quantization']
    data_type = interpreter.get_input_details()[0]['dtype']

    img = cv2.imread(img_dir)
    img_sqr = sqr_crop(img)
    img_resized = cv2.resize(img_sqr, (input_height, input_width))
    frame = img_resized
    img_scaled = (img_resized / input_scale) + input_zero_point
    img_expanded = np.expand_dims(img_scaled, axis=0).astype(data_type)

    cv2.imshow("image", img_resized)
    cv2.waitKey(1000)
    
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_expanded)
    interpreter.invoke()
    
#    model_output_data_0 = interpreter.get_tensor(interpreter.get_output_details()[0]['index']) / 1
#    model_output_data_1 = interpreter.get_tensor(interpreter.get_output_details()[1]['index']) / 1
#    model_output_data_2 = interpreter.get_tensor(interpreter.get_output_details()[2]['index']) / 1
#    model_output_data_3 = interpreter.get_tensor(interpreter.get_output_details()[3]['index']) / 1

    model_output_data_0 = (interpreter.get_tensor(interpreter.get_output_details()[0]['index']) - input_zero_point) * input_scale
    model_output_data_1 = (interpreter.get_tensor(interpreter.get_output_details()[1]['index']) - input_zero_point) * input_scale
    model_output_data_2 = (interpreter.get_tensor(interpreter.get_output_details()[2]['index']) - input_zero_point) * input_scale
    model_output_data_3 = (interpreter.get_tensor(interpreter.get_output_details()[3]['index']) - input_zero_point) * input_scale
     
#    grasp_positions_out = (model_output_data_0 - input_zero_point) * input_scale
#    grasp_angles_out = np.arctan2(model_output_data_2, model_output_data_1)/2.0
#    grasp_width_out = (model_output_data_3 - input_zero_point) * input_scale

    grasp_positions_out = model_output_data_0
    grasp_angles_out = np.arctan2(model_output_data_2, model_output_data_1)/2.0
    grasp_width_out = model_output_data_3
    
    
    #### finding grasp properties from model output ####
    # peak_local_max returns a list of all peaks, need to add loop through all peaks

    # get grasp postion
    grasp_position_img = grasp_positions_out[0, ].squeeze()
    grasp_position_img = gaussian(grasp_position_img, 0, preserve_range=True)
    local_max = peak_local_max(grasp_position_img, min_distance=20, threshold_abs=0.2, num_peaks=1) # very slow, library problem?
    local_max = local_max.squeeze()
    print(local_max)
    # local_max = [local_max[0]//(input_width_g//input_width_g), local_max[1]//(input_width_g//input_width_g)]

    if len(local_max):
         # get grasp width
         grasp_width_img = grasp_width_out[0, ].squeeze()
         grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)
         grasp_l = grasp_width_img[tuple(local_max)]
         grasp_w = grasp_l/2.0

         # get grasp angle
         grasp_angles_img = grasp_angles_out[0, ].squeeze()
         grasp_point = tuple(local_max)
         grasp_angle = grasp_angles_img[grasp_point]
         ang_threshold = 5

    if grasp_angle > 0:
        grasp_angle = grasp_angles_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                         grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].max()
    else:
        grasp_angle = grasp_angles_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                         grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].min()

     # converting grasp to line
    line_x1 = int(local_max[1]-(grasp_w*math.sin(grasp_angle)))
    line_y1 = int(local_max[0]+(grasp_w*math.cos(grasp_angle)))
    line_x2 = int(local_max[1]+(grasp_w*math.sin(grasp_angle)))
    line_y2 = int(local_max[0]-(grasp_w*math.cos(grasp_angle)))

    # show grasp on frame
    cv2.circle(frame, (local_max[1], local_max[0]), 2, (0,255,0), 1)
    cv2.line(frame, (line_x1,line_y1), (line_x2,line_y2), (255, 0, 0), 2)

    frame = cv2.resize(frame, (600, 600))
    cv2.imshow("Frame", frame)
    cv2.waitKey(1000)

if __name__ == '__main__':
    main()
