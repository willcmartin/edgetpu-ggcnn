"""
Source: https://github.com/dougsm/ggcnn

Source license: BSD 3-Clause "New" or "Revised" License

Copyright (c) 2018, Douglas Morrison, ARC Centre of Excellence for Robotic Vision (ACRV), Queensland University of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
from datetime import datetime

import h5py
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input
from keras.models import Model


FILTER_SIZES = [
    [(9, 9), (5, 5), (3, 3)]
]

NO_FILTERS = [
    [32, 16, 8],
]

INPUT_DATASET = 'data/datasets/dataset_210516_1505.hdf5'

# =====================================================================================================
# Load the data.
f = h5py.File(INPUT_DATASET, 'r')

rgb_train = np.array(f['train/rgb'])
point_train = np.expand_dims(np.array(f['train/grasp_points_img']), -1)
angle_train = np.array(f['train/angle_img'])
cos_train = np.expand_dims(np.cos(2*angle_train), -1)
sin_train = np.expand_dims(np.sin(2*angle_train), -1)
grasp_width_train = np.expand_dims(np.array(f['train/grasp_width']), -1)

rgb_test = np.array(f['test/rgb'])
point_test = np.expand_dims(np.array(f['test/grasp_points_img']), -1)
angle_test = np.array(f['test/angle_img'])
cos_test = np.expand_dims(np.cos(2*angle_test), -1)
sin_test = np.expand_dims(np.sin(2*angle_test), -1)
grasp_width_test = np.expand_dims(np.array(f['test/grasp_width']), -1)

# Ground truth bounding boxes.
gt_bbs = np.array(f['test/bounding_boxes'])

f.close()

# ====================================================================================================
# Set up the train and test data.

x_train = rgb_train

grasp_width_train = np.clip(grasp_width_train, 0, 150)/150.0
y_train = [point_train, cos_train, sin_train, grasp_width_train]

x_test = rgb_test

grasp_width_test = np.clip(grasp_width_test, 0, 150)/150.0
y_test = [point_test, cos_test, sin_test, grasp_width_test]

# ======================================================================================================================

for filter_sizes in FILTER_SIZES:
    for no_filters in NO_FILTERS:
        dt = datetime.now().strftime('%y%m%d_%H%M')

        NETWORK_NAME = "ggcnn_%s_%s_%s__%s_%s_%s" % (filter_sizes[0][0], filter_sizes[1][0], filter_sizes[2][0],
                                                     no_filters[0], no_filters[1], no_filters[2])
        NETWORK_NOTES = """
            Input: Inpainted depth, subtracted mean, in meters, with random rotations and zoom.
            Output: q, cos(2theta), sin(2theta), grasp_width in pixels/150.
            Dataset: %s
            Filter Sizes: %s
            No Filters: %s
        """ % (
            INPUT_DATASET,
            repr(filter_sizes),
            repr(no_filters)
        )
        OUTPUT_FOLDER = 'data/networks/%s__%s/' % (dt, NETWORK_NAME)

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        # Save the validation data so that it matches this network.
        np.save(os.path.join(OUTPUT_FOLDER, '_val_input'), x_test)

        # ====================================================================================================
        # Network

        input_layer = Input(shape=x_train.shape[1:])

        x = Conv2D(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu')(input_layer)
        x = Conv2D(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu')(x)
        encoded = Conv2D(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu')(x)

        x = Conv2DTranspose(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu')(encoded)
        x = Conv2DTranspose(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2DTranspose(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu')(x)

        # ===================================================================================================
        # Output layers

        pos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='pos_out')(x)
        cos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='cos_out')(x)
        sin_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='sin_out')(x)
        width_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='width_out')(x)

        # ===================================================================================================
        # And go!

        ae = Model(input_layer, [pos_output, cos_output, sin_output, width_output])
        ae.compile(optimizer='rmsprop', loss='mean_squared_error')

        ae.summary()

        with open(os.path.join(OUTPUT_FOLDER, '_description.txt'), 'w') as f:
            # Write description to file.
            f.write(NETWORK_NOTES)
            f.write('\n\n')
            ae.summary(print_fn=lambda q: f.write(q + '\n'))

        with open(os.path.join(OUTPUT_FOLDER, '_dataset.txt'), 'w') as f:
            # Write dataset name to file for future reference.
            f.write(INPUT_DATASET)

        tb_logdir = './data/tensorboard/%s_%s' % (dt, NETWORK_NAME)

        my_callbacks = [
            TensorBoard(log_dir=tb_logdir),
            ModelCheckpoint(os.path.join(OUTPUT_FOLDER, 'epoch_{epoch:02d}_model.hdf5'), period=1),
        ]

        ae.fit(x_train, y_train,
               batch_size=4,
               epochs=50,
               shuffle=True,
               callbacks=my_callbacks,
               validation_data=(x_test, y_test)
               )
