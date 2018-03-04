"""Main: Classify structural building damage from images of the exterior"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import math
import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from collections import namedtuple

tf.logging.set_verbosity(tf.logging.INFO)

MY_DATA_FOLDER = 'data'

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    """Input Layer, i.e., the features or pixels of the image, reshaped to be a
    128x128 pixel image with a batch_size of `-1` which indicates the
    batch_size should be automatically determined. The number of channels is
    1, i.e., the number of color channels (in this case, img is grayscale)"""
    input_layer = tf.reshape(
        features["x"],  # data to reshape; our image features
        [-1, 128, 128, 1] # desired shape: [batch_size, width, height, channels]
    )

    # First Convolutional Layer; output shape = [batch_size, 128, 128, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,    # inputs from the previous layer of features
        filters=32,            # a total of 32 filters, creating 32 outputs
        kernel_size=[3, 3],    # 3x3 convolution tiles
        padding="same",        # output padded to have same dimensions as input
        activation=tf.nn.relu  # ReLU activation applied to the output
    )

    # First Pooling Layer; output shape = [batch_size, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,          # inputs from the previous convolutional layer
        pool_size=[2, 2],      # max_pool each 2x2 square into 1 value
        strides=2              # stride 2 squares to the right after each pool
    )

    # Second Conv & Pooling Layers; output shape = [batch_size, 32, 32, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,            # now using 64 filters, creating 64 outputs
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # max_pooling reduces our `image` width and height by 50%
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,            # now using 64 filters, creating 64 outputs
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # max_pooling reduces our `image` width and height by 50%
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Flatten pool2 for input to dense layer; out_shape=[batch_size, 16,384]
    pool3_flat = tf.reshape(pool3, [-1, 16 * 16 * 64])  # reshape to be flat (2D)

    # Performs the actual classification of the abstracted features from conv1/2
    dense = pool3_flat
    for i in range(3):
        dense = tf.layers.dense(
            inputs=dense,     # inputs from flattened pool layer
            units=4096 // 4**i,  # number of neurons
            activation=tf.nn.relu  # ReLU activation function
        )

    # Dropout creates a chance that input is ignored during training. This will
    # decrease the chances of over-fitting the training data
    dropout = tf.layers.dropout(
        inputs=dense,          # inputs from the dense layer
        rate=0.5,              # randomly drop-out 40% of samples; keep 60%
        training=(mode == tf.estimator.ModeKeys.TRAIN)  # is training mode? T/F
    )
    # Output of dense layer, shape = [batch_size, 256]

    """Final, logits layer giving us our outputs as probabilities of the original
    input image being a undamaged building (0) or a collapsed building (1). Thus,
    we have a vector of 2 outputs which will each represent the probability for
    the input to be categorized with one of the two types of structural damage.
    If the first value, output[0] is high, then we say that the model predicts
    a high chance that the input was an undamaged building. This final layer
    is densely connected to the previous layer and provides 2 outputs. These
    outputs will later be turned into values from 0 to 1 representing actual
    probabilities (see predictions layer and tf.nn.softmax() below)
    Final shape: [batch_size, 2]"""
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Predicted class determined by finding the maximum value of the logits
        "classes": tf.argmax(input=logits, axis=1),  # get the predicted class

        # We get the normalize the probabilities to be from 0 to 1 with the sum
        # of both equaling 1.0 by performing a softmax on the output vector.
        "probabilities": tf.nn.softmax(
            logits,                # softmax on logits (output of CNN)
            name="softmax_tensor"  # name for logging
        )
    }

    """If we are in prediction mode, return a predictor tensor of the network
    and stop; no more building required for prediction"""
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    """For TRAIN and EVAL modes, calculate the loss. Cross entropy is a loss
    technique commonly used when we have many classes to classify. This is
    just another function that calculates the error between our predicted
    class and the actual target class. The more wrong we are, the greater the
    loss is. e.g., predict 10% chance for a 0 and it is a 0 is bad; predict
    90% chance for a 0 and it is a 0 is better. Cross entropy is only
    concerned with the prediction accuracy for the target value."""
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,  # labels are the actual truth values of the data
        logits=logits   # logits are our predicted values of the data
    )

    # Configure the training operation if TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        # We want a low learning rate so that we can slowly but surely reach
        # the optimum. Higher rates may learn faster but may overshoot the opt
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004)

        # Use our Grad Descent optimizer to minimize the loss we calculated!
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        # return our estimator with the loss and train_op to train the model
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    # Add evaluation metrics for EVAL phase
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,  # true labels given in data set
            predictions=predictions["classes"]  # predicted classes found above
        )
    }

    # return our estimator with the loss and eval_metric_ops to find accuracy
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )

def main(_):
    # Load custom, hand-labeled training and eval data
    dataset = load_dataset()
    train_data = dataset.train.images
    train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
    eval_data = dataset.test.images
    eval_labels = np.asarray(dataset.test.labels, dtype=np.int32)

    # Create the actual Estimator to run our model
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,  # the CNN model we created earlier!
        model_dir="./convnet_quake"  # save model data here
    )

    # Setup logging here
    tensors_to_log = {"probabilities": "softmax_tensor"} # prob from earlier
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,  # format {name-for-log: tensor-to-log}
        every_n_iter=100  # log every 50 steps of training
    )

    # Training the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(  # org. our inputs
        x={"x": train_data},  # set the feature data
        y=train_labels,       # set the truth labels
        batch_size=len(train_data),
        num_epochs=None,
        shuffle=True)         # randomize
    classifier.train(
        input_fn=train_input_fn,  # training inputs organized above
        steps=500,
        hooks=[logging_hook])     # connect to logging

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    # evaluate based on the data!
    eval_results = classifier.evaluate(
        input_fn=eval_input_fn,
        hooks=[logging_hook]
    )

    print(eval_results)  # Print our results and accuracy!

def load_dataset():
    """Load our train/test data and attach labels."""

    train_filepath = os.path.join('data', 'train', 'proc')
    test_filepath = os.path.join('data', 'test', 'proc')

    def getFeatureSet(data_filepath):
        """Return a namedtuple with a set of images and labels."""

        FeatureSet = namedtuple('FeatureSet', ['images', 'labels'])

        images = []
        labels = []
        for label in os.listdir(data_filepath):
            for file in os.listdir(os.path.join(data_filepath, label)):

                # Read grayscale image, flatten and normalize, then add label
                img = cv2.imread(os.path.join(data_filepath, label, file), cv2.IMREAD_GRAYSCALE)
                images.append(img.flatten() / 255.0)
                labels.append(int(label))

        return FeatureSet(
            images=np.array(images, dtype=np.float32),
            labels=np.array(labels, dtype=np.float32)
        )

    return namedtuple('Data', ['train', 'test'])(
        train=getFeatureSet(train_filepath),
        test=getFeatureSet(test_filepath)
    )

if __name__ == "__main__":
  tf.app.run()
