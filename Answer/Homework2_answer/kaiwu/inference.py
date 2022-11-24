import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import ops


class Inference:
    def __init__(self, alg, saver=None) -> None:
        self.alg = alg
        cpu_num = 1
        self.sess_config = tf.ConfigProto(
            device_count={"CPU": cpu_num},
            inter_op_parallelism_threads=cpu_num,
            intra_op_parallelism_threads=cpu_num,
            log_device_placement=False,
        )
        self.saver = saver

    @property
    def graph(self):
        return self.alg.get_graph()

    @graph.setter
    def graph(self, graph):
        self.alg.set_graph(graph)

    def init(self):
        self.alg.build_infer_graph()
        if self.saver is None:
            self.saver = tf.train.Saver(
                self.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES),
                allow_empty=True,
            )

    def run(self, feed_dict=None, checkpint_path=None, init=False):

        if feed_dict is None:
            feed_dict = self.alg.random_data()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            if init:
                sess.run(tf.global_variables_initializer())
            if checkpint_path:
                self.restore_model(sess, checkpint_path)
            return sess.run(self.alg.get_output_tensors(), feed_dict=feed_dict)

    def generate_random_checkpoint_pb(self, directory="checkpoints"):
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            checkpint_path = self.save_checkpoint(sess, directory=directory)
            print("-" * 10, "save random checkpoint to", checkpint_path)
            pb_filepath = self.save_as_pb(checkpint_path, directory=directory)
            print("-" * 10, "save pb", pb_filepath)

    def save_checkpoint(
        self, sess, directory="checkpoints", filename="model.ckpt"
    ):
        os.makedirs(directory, exist_ok=True)
        checkpint_path = os.path.join(directory, filename)
        return self.saver.save(sess, checkpint_path)

    def restore_model(self, sess, checkpint_path="checkpoints"):
        ckpt = tf.train.get_checkpoint_state(checkpint_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_as_pb(self, checkpint_path, directory="checkpoints", filename="frozen"):
        os.makedirs(directory, exist_ok=True)

        pbtxt_filename = filename + ".pbtxt"
        pbtxt_filepath = os.path.join(directory, pbtxt_filename)
        pb_filepath = os.path.join(directory, filename + ".pb")

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.write_graph(
                graph_or_graph_def=sess.graph_def,
                logdir=directory,
                name=pbtxt_filename,
                as_text=True,
            )

        freeze_graph.freeze_graph(
            input_graph=pbtxt_filepath,
            input_saver="",
            input_binary=False,
            input_checkpoint=checkpint_path,
            output_node_names=",".join(
                [t.op.name for t in self.alg.get_output_tensors()]
            ),
            restore_op_name="Unused",
            filename_tensor_name="Unused",
            output_graph=pb_filepath,
            clear_devices=True,
            initializer_nodes="",
        )

        return pb_filepath

    def load_from_pb(self, pb_filepath):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_filepath, "rb") as f:
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(
                graph_def,
                name="",
            )

        output_tensors = []
        input_tensors = []
        for op in self.graph.get_operations():
            if "exp_output" in op.name:
                for o in op.outputs:
                    output_tensors.append(self.graph.get_tensor_by_name(o.name))
            if "Placeholder" in op.name:
                for o in op.outputs:
                    input_tensors.append(self.graph.get_tensor_by_name(o.name))

        self.alg.set_input_tensor(input_tensors)
        self.alg.set_output_tensor(output_tensors)
        self.saver = tf.train.Saver(
            self.graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES),
            allow_empty=True,
        )

    def random_data(self):
        return self.alg.random_data()


def cmd_init(exp):
    if exp == "exp_1":
        from exp_1 import Algorithm
    elif exp == "exp_2":
        from exp_2 import Algorithm
    else:
        raise Exception("Unknown exp: {}".format(exp))

    infer = Inference(Algorithm())
    infer.init()
    infer.generate_random_checkpoint_pb(exp)


def usage(argv):
    print("-" * 10, "Usage: python3 {} init <exp_1|exp_2>".format(argv[0]))


def main(argv):
    tf.logging.set_verbosity(tf.logging.ERROR)
    if len(argv) < 3:
        usage(argv)
        return

    cmd = argv[1]
    if cmd == "init":
        cmd_init(argv[2])
    else:
        raise Exception("Unknown argv: {}".format(argv))


if __name__ == "__main__":
    import sys

    main(sys.argv)
