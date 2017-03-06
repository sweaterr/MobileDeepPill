import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
import numpy as np

class FeatureGenerator:

    sess = None
    graph_pb_path = None

    def __init__(self, model_path, input_size):

        self.graph_pb_path = model_path
        self.input_size = input_size


        with tf.Graph().as_default():
            # obtain the handler of current graphdef
            output_graph_def = graph_pb2.GraphDef()

            # import the graph pb
            with open(self.graph_pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = importer.import_graph_def(output_graph_def, name="")

            self.sess = tf.Session()


    def gen_feature(self, imgs):
        assert len(imgs.shape) == 4
        assert imgs.shape[1:] == (self.input_size,self.input_size,3)
        color_fea_node = self.sess.graph.get_tensor_by_name("color_fea:0")
        gray_fea_node = self.sess.graph.get_tensor_by_name("gray_fea:0")
        color_ph = self.sess.graph.get_tensor_by_name("color_ph:0")
        color_fea, gray_fea = self.sess.run([color_fea_node, gray_fea_node], feed_dict={color_ph: imgs})
        return color_fea, gray_fea



    def close(self):

        if self.sess is not None:
            self.sess.close()

