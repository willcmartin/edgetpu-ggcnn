# Generative Grasping CNN (GG-CNN) implemented on Edge TPU

## Goal
Train and minimally implement the GG-CNN model on an Edge TPU

## Credit
Credit for the model and much of the basic code goes to: https://github.com/dougsm/ggcnn

## Instructions
1. Train
2. Convert
3. Evaluate

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
