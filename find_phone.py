r"""Display center coordinates of detected phone in input image.

Module adapted by Robin Laird from TensorFlow Object Detection API tutorial.
The original file is nominally located at:

/models/research/object_detection/object_detection_tutorial.ipynb

It has been modified to process a single image using a frozen inference graph
that has been trained using image data containing a single object class.

Example usage:
    python find_phone.py image_path

Assumes:
    Frozen inference graph file is in current directory.
    Version v1.4.* of TensorFlow is installed.
"""

# Script dependencies.
import os
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image
from object_detection.utils import ops as utils_ops

# Model information - assume script is run from dir in which model data is held.
PHONE_INFERENCE_GRAPH_PATH = "phone_graph.pb"


def find_center(x1, y1, x2, y2):
    # Determines x,y location of center of rectangle defined by x1,y1 and x2,y2.
    # args:
    #   x1, y1: "upper left" corner of rectangle
    #   x2, y2: "lower right" corner of rectangle
    # returns:
    #   x,y coordinates of center of the rectangle
    center_x = (x1 + x2)/2
    center_y = (y1 + y2)/2
    return center_x, center_y


def load_image_into_numpy_array(image):
    # Converts in-memory image into np.array to facilitate further processing.
    # args:
    #   image: image to convert into array
    # returns:
    #   np.array containing input image
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    # Runs a detection on a single image using the input graph.
    # args:
    #   image: input image to process (detect objects in)
    #   graph: frozen graph through which to process image
    # returns:
    #   dictionary containing num detections and detection info for each detection
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors.
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                "num_detections", "detection_boxes", "detection_scores",
                "detection_classes", "detection_masks"
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if "detection_masks" in tensor_dict:
                # The following processing is only for single image.
                detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
                detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
                # Re-frame is required to translate mask from box coordinates to image coordinates
                # and fit the image size.
                real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension.
                tensor_dict["detection_masks"] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")
            # Run inference.
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            # All outputs are float32 numpy arrays, so convert types as appropriate.
            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]
            if "detection_masks" in output_dict:
                output_dict["detection_masks"] = output_dict["detection_masks"][0]
    return output_dict


def main(_):
    # Check TensorFlow version - require at least v1.4.
    if tf.__version__ < "1.4.0":
        print("Please upgrade your TensorFlow installation to v1.4.* or later, exiting.")
        exit(1)

    # Construct argument parser and parse arguments.
    ap = argparse.ArgumentParser(description="Display center coordinates of detected phone in input image.")
    ap.add_argument("image", help="path to phone image file")
    args = ap.parse_args()
    image_path = args.image

    # See if we can find the image file.
    if not os.path.exists(image_path):
        print("Cannot locate the specified image file, exiting.")
        exit(1)

    # See if we can find the frozen tensor model file.
    if not os.path.exists(PHONE_INFERENCE_GRAPH_PATH):
        print("Cannot locate the phone inference graph file, exiting.")
        exit(1)

    # Load frozen tensor model (classifier) into memory.
    # Currently using "faster_rcnn_resnet101_coco_2018_01_28".
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PHONE_INFERENCE_GRAPH_PATH, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Attempt to open image.
    image = Image.open(image_path)
    # Convert image into array.
    image_np = load_image_into_numpy_array(image)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Calculate centroid of detection box.
    # Detection box is: ymin[0][0], xmin[0][1], ymax[0][2], xmax[0][3]
    if output_dict["detection_boxes"] is not None:
        x, y = find_center(output_dict["detection_boxes"][0][1],  # x1
                           output_dict["detection_boxes"][0][0],  # y1
                           output_dict["detection_boxes"][0][3],  # x2
                           output_dict["detection_boxes"][0][2])  # y2
        print("%0.4f %0.4f" % (x, y))


if __name__ == "__main__":
    tf.app.run()
