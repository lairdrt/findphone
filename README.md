# findphone
Find phone in an image and determine the normalized coordinates of its center within the image.
## Background
Classic localization problem given an untagged set of images and a set of labels identifying the normalized center point of the phone in the image. Task was to localize the phone within each image and report its center point in normalized coordinates. Employed transfer learning using frozen inference graph based upon the Faster RCNN RESNET model (faster_rcnn_resnet101_coco_2018_01_28) from the TensorFlow detection model zoo.
## Overall Task
### Step 1: Train the Model on Phone Images
This was accomplished using the TensorFlow Object Detection API, and utility software supplied with the API. Specifically, this was done by:
1. Installing object detection API.
2. Installing labelImg software.
3. Tagging phones in test images using labelImg, which saves bounding box in Pascal VOC XML format.
4. Creating TensorFlow Record (tfr) from Pascal VOC XML files using modified version of create_pet_tf_record.py from the object detection API (located under cloned API dir at ./tensorflow/models/research/object_detection/dataset_tools). Got caught up here not realzing that, in create_pet_tf_record.py, the class name of the object to detect was encoded in the filename. Hardcoded this to specify the class (i.e., "cellphone") of interest.
5. Installing the Faster RCNN RESNET model from the zoo.
6. Modifying pipeline.config to minimize batch sizes in the train.config section (batch_queue_capacity and prefetch_queue_capacity set to 4). I have a very old Dell Dimension E520 with little memory and the training would stall without these changes.
7. Training on the sample images for 8+ hours until the total loss was less than ~0.05. Ended up with a checkpoint in the 254 range.
## References
TensorFlow detection model zoo:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
TensorFlow object detection API:  
https://github.com/tensorflow/models/tree/master/research/object_detection  
LabelImage software:  
https://github.com/saicoco/object_labelImg  
