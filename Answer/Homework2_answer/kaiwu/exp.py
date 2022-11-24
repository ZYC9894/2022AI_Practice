import numpy as np
import tensorflow as tf


class AlgorithmBase:
    def __init__(self) -> None:
        self.input_tensors = []
        self.output_tensors = []
        self.graph = None

    def set_graph(self, graph):
        self.graph = graph

    def get_graph(self):
        return self.graph

    def set_input_tensor(self, input_tensors):
        self.input_tensors = input_tensors

    def set_output_tensor(self, output_tensors):
        self.output_tensors = output_tensors

    def get_output_tensors(self):
        return self.output_tensors

    def get_input_tensors(self):
        return self.input_tensors

    def random_data(self):
        feed_dict = {}
        for tensor in self.get_input_tensors():
            feed_dict[tensor.name] = np.random.rand(
                *[dim.value for dim in tensor.shape.dims]
            )
        return feed_dict

    def _build_input_tensors(self):
        raise Exception("No implementation")

    def _build_output_tensor(self):
        raise Exception("No implementation")

    def build_infer_graph(self):
        self.graph = tf.Graph()
        self.output_tensors = []
        self.input_tensors = []

        with self.graph.as_default():
            self._build_input_tensors()

            output_tensors = self._build_output_tensor()
            with tf.name_scope("exp_output"):
                self.set_output_tensor(
                    [tf.identity(tensor) for tensor in output_tensors]
                )
