
# Grape Detection and Segmentation

This is an example showing the use of Mask RCNN on grape detection and segmentation。We reference [the paper](https://arxiv.org/abs/1907.11819) and use its grape database ([WGISD](https://github.com/thsant/wgisd): Embrapa Wine Grape Instance Segmentation Dataset )。First, we build our grape dataset for Mask RCNN on WGISD. Then we train and evaluate the model. At last, we provide two ways of visualizing the grape detection and segmentation.

The code has been tested on Python 3.7.3,Tensorflow 1.14.0,Keras 2.2.4,and [Mask_RCNN](https://github.com/matterport/Mask_RCNN).

<p align="Center">
  <img src="assets/show/splash.jpg" width="350" title="hover text">
  <img src="assets/show/detect.jpg.jpg" width="350" title="hover text">
</p>