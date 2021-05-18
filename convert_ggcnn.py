import tensorflow as tf
# from tensorflow import keras
import keras
import h5py
import numpy as np

# load dataset
dataset_fn = '/Users/willmartin/work/CAR_lab/edgetpu-ggcnn/data/datasets/dataset_210516_1505.hdf5'
f = h5py.File(dataset_fn, 'r')

# rgb_imgs = np.array(f['test/rgb']) # remove if below line works
rep_imgs = np.array(f['test/rgb'], dtype=np.float32)

#representative dataset
def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((rep_imgs)).batch(1).take(150):
      yield [data]

# load model
model_checkpoint_fn = '/Users/willmartin/work/CAR_lab/edgetpu-ggcnn/data/networks/210516_1705__ggcnn_9_5_3__32_16_8/epoch_50_model.hdf5'
model = keras.models.load_model(model_checkpoint_fn)
model.summary()

# convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
converter.experimental_new_converter = False
tflite_quant_model = converter.convert()
