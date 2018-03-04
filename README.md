# quake-triage
Using deep learning to triage earthquake damage

## What is this?
We created a neural network to analyze photos of buildings and triage the level of
structural damage. Our approach used a convolutional neural network with 3 convolutions
(with max-pooling) and 4 dense (fully-connected) layers. The output is a classification
result that indicates a fully intact building (0) or completely collapsed building (1).

For next steps, we plan to convert this model to regression which will let us assess
the actual severity of the damage. Additionally, the amount of noise in the images
(e.g., trees, pedestrians) negatively affects classification accuracy since this noise
does not contribute to the determination of damage of the building. Therefore, we would
seek out ways to filter out or remove this noise from our image before classification.

## Why is this useful?
Assessing the structural damage of buildings automatically is very important, especially
after disaster events. In a disaster scenario (e.g., earthquake), our first response
would be to collect images of buildings in the affected area in two ways. First, we could
fly a drone over the area and capture photos of affected buildings. Also, we could leverage
social media and crowdsource images of damage from posts in the area. Then, this data
could be aggregated and used to create a heatmap of the affect area due to the disaster
which could then inform responders on how to best treat the area. Our map would locate
the areas with the most damage and help assess the cost of repairs and assistance.

## Workspace
+ `/data` - Train/test data sorted by labels stored here.
+ `/convnet_quake` - Created by Tensorflow after running `main.py`.

## Run
+ Download the following:
 + [`numpy`](http://www.numpy.org/)
 + [`tensorflow`](https://www.tensorflow.org/)
 + [`scipy`](https://www.scipy.org/)
 + [`opencv-python`](https://pypi.python.org/pypi/opencv-python)
