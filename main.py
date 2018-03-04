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
    256x256 pixel image with a batch_size of `-1` which indicates the
    batch_size should be automatically determined. The number of channels is
    3, i.e., the number of color channels (in this case, img is RGB)"""
    input_layer = tf.reshape(
        features["x"],  # data to reshape; our image features
        [-1, 256, 256, 3] # desired shape: [batch_size, width, height, channels]
    )

    # First Convolutional Layer; output shape = [batch_size, 256, 256, 16]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,    # inputs from the previous layer of features
        filters=32,            # a total of 32 filters, creating 32 outputs
        kernel_size=[5, 5],    # 5x5 convolution tiles
        padding="same",        # output padded to have same dimensions as input
        activation=tf.nn.relu  # ReLU activation applied to the output
    )

    # First Pooling Layer; output shape = [batch_size, 128, 128, 16]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,          # inputs from the previous convolutional layer
        pool_size=[2, 2],      # max_pool each 2x2 square into 1 value
        strides=2              # stride 2 squares to the right after each pool
    )

    # Second Conv & Pooling Layers; output shape = [batch_size, 64, 64, 32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,            # now using 64 filters, creating 64 outputs
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # max_pooling reduces our `image` width and height by 50%
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Third Conv & Pooling Layers; output shape = [batch_size, 32, 32, 64]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,            # now using 64 filters, creating 64 outputs
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # max_pooling reduces our `image` width and height by 50%
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Flatten pool2 for input to dense layer; out_shape=[batch_size, 3136]
    pool3_flat = tf.reshape(pool3, [-1, 32 * 32 * 3 * 64])  # reshape to be flat (2D)

    # Performs the actual classification of the abstracted features from conv1/2
    dense = tf.layers.dense(
        inputs=pool3_flat,     # inputs from flattened pool layer
        units=512,             # 1,024 neurons in the layer (1024 outputs)
        activation=tf.nn.relu  # ReLU activation function
    )

    # Dropout creates a chance that input is ignored during training. This will
    # decrease the chances of over-fitting the training data
    dropout = tf.layers.dropout(
        inputs=dense,          # inputs from the dense layer
        rate=0.2,              # randomly drop-out 40% of samples; keep 60%
        training=(mode == tf.estimator.ModeKeys.TRAIN)  # is training mode? T/F
    )
    # Output of dense layer, shape = [batch_size, 1024]

    """Final, logits layer giving us our outputs as probabilities of the original
    input image being a digit 0-9. Thus, we have a vector of 10 outputs which
    will each represent the probability for the input to be the digit. e.g.,
    if the first value, output[0] is high, then we say that the model predicts
    a high chance that the input was a handwritten digit '0'. This final layer
    is densely connected to the previous layer and provides 10 outputs. These
    outputs will later be turned into values from 0 to 1 representing actual
    probabilities (see predictions layer and tf.nn.softmax() below)
    Final shape: [batch_size, 10]"""
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Predicted class determined by finding the maximum value of the logits
        "classes": tf.argmax(input=logits, axis=1),  # get the predicted class

        # We get the normalize the probabilities to be from 0 to 1 with the sum
        # of all 10 equaling 1.0 by performing a softmax on the output vector.
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
    loss is. e.g., predict 10% chance for a 5 and it is a 5 is bad; predict
    90% chance for a 5 and it is a 5 is better. Cross entropy is only
    concerned with the prediction accuracy for the target value."""
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,  # labels are the actual truth values of the data
        logits=logits   # logits are our predicted values of the data
    )

    # Configure the training operation if TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        # We want a low learning rate so that we can slowly but surely reach
        # the optimum. Higher rates may learn faster but may overshoot the opt
        optimizer = tf.train.AdamOptimizer(learning_rate=0.06)

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
    # Load training and eval data
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
        every_n_iter=50  # log every 50 steps of training
    )

    # Training the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(  # org. our inputs
        x={"x": train_data},  # set the feature data
        y=train_labels,       # set the truth labels
        batch_size=8,       # num samples to give at a time - orig: 100
        num_epochs=None,
        shuffle=True)         # randomize
    classifier.train(
        input_fn=train_input_fn,  # training inputs organized above
        steps=500,                # orig: 20000 training steps
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
    FeatureSet = namedtuple('FeatureSet', ['images', 'labels'])

    train_filepath = os.path.join('data', 'train', 'proc')
    test_filepath = os.path.join('data', 'test', 'proc')

    images = []
    labels = []
    for label in os.listdir(train_filepath):
        for file in os.listdir(os.path.join(train_filepath, label)):
            img = cv2.imread(os.path.join(train_filepath, label, file))
            images.push(img.flatten() / 255.0)
            labels.push(label)

    trainSet = FeatureSet(
        images=images,
        labels=labels
    )

    images = []
    labels = []
    for label in os.listdir(test_filepath):
        for file in os.listdir(os.path.join(test_filepath, label)):
            img = cv2.imread(os.path.join(test_filepath, label, file))
            images.push(img.flatten() / 255.0)
            labels.push(label)

    testSet = FeatureSet(
        images=images,
        labels=labels
    )

    return namedtuple('Data', ['train', 'test'])(
        train=trainSet,
        test=testSet
    )
>>>>>>> Stashed changes

if __name__ == "__main__":
  tf.app.run()
