# Generative Grasping CNN (GG-CNN) Implemented on Edge TPU

## Goal
Train and minimally implement the GG-CNN model on an Edge TPU

## Credit
Credit for the model and most of the code goes to: https://github.com/dougsm/ggcnn

## Procedure
1. Generate Dataset
    - Download
        - [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)
        - The offical dataset is [here](pr.cs.cornell.edu/grasping/rect_data/data.php), but the website has been down for awhile
        - Eventually I would like to use the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/)
    - Place dataset files in data/cornell
    - Run generate_dataset.py
        - This will create the needed .hdf5 file in data/datasets
2. Train
    - Run train_ggcnn.py
    - TensorFlow model files for each epoch will be create in data/networks
3. Convert
    -  
4. Evaluate

Currently implemented:
- Training
    - Using Google Colab with free GPU
    - Cornell dataset
- Post-traning quantization
- Compliling model for Edge TPU
- Model testing with Google Colab and python file

TODO:
- Quantization aware training
- Jaquard dataset

General info:
- Cornell grasping dataset recieved from Kaggle

Goal:
- Get grasp returns from an image of any size on coral tpu
- Image: RGB, any size
- Train that model from at least Cornell dataset
- Display grasps on an image
