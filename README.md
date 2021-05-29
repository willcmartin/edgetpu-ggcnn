# GG-CNN implemented on Edge TPU
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
- Add file for generating cornell dataset

General info:
- Cornell grasping dataset recieved from Kaggle

Goal:
- Get grasp returns from an image of any size on coral tpu
- Image: RGB, any size
- Train that model from at least Cornell dataset
- Display grasps on an image
