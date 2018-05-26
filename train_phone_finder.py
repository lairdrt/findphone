r"""Stub for model training that was done off-line.

Example usage:
    python train_phone_finder.py image_folder_path

Assumes:
    Inference graph was produced off-line.
    Compressed model file is in current path.
"""

import tensorflow as tf
import tarfile

def main(_):
    # Extract compressed inference graph.
    print("Phone inference graph created using faster_rcnn_resnet101_coco_2018_01_28 model.")
    tar = tarfile.open("phone_graph.tar.gz")
    tar.extractall()
    tar.close()

if __name__ == "__main__":
    tf.app.run()
