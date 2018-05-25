# findphone
Find phone in an image and determine the normalized coordinates of its center within the image.
# Background
Classic localization problem given an untagged set of images and a set of labels identifying the normalized center point. Task is to localize the phone within each image and report its center point in normalized coordinates. Employed transfer learning using frozen inference graph based upon the Faster RCNN RESNET model (101 faster_rcnn_resnet101_coco_2018_01_28 from the TensorFlow detection model zoo.
# Overall Task
## Step 1: Train the Model on Phone Images
This was accomplished "off-line" using the TensorFlow Object Detection API, and utility software supplied with the API. Specifically
# References
TensorFlow detection model zoo:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
TensorFlow object detection API:  
https://github.com/tensorflow/models/tree/master/research/object_detection  
