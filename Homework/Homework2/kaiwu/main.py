import os
import numpy as np

from exp import AlgorithmBase
from inference import Inference


def get_inference(alg):
    infer = Inference(alg)
    infer.init()
    return infer


def get_base_inference(pb_filepath, alg):
    trainer = Inference(alg)
    trainer.load_from_pb(pb_filepath)
    return trainer


def run_test(times, infer, infer_base, checkpoint_path):

    for i in range(times):
        feed_dict = infer.random_data()
        got = infer.run(feed_dict, checkpoint_path)
        expected = infer_base.run(feed_dict, checkpoint_path)

        if len(got) != len(expected):
            print("Test failed: len(got) != len(expected)", got, expected)
            break
        same = True
        for j in range(len(got)):
            if not (np.array(got[j]) - np.array(expected[j]) < 1e-5).all():
                same = False
                break

        if same:
            print("-" * 10, "Test", i, "passed")
        else:
            print(
                "-" * 10,
                "Test",
                i,
                "failed",
                "\n",
                "-" * 10,
                "got",
                got,
                "\n",
                "-" * 10,
                "expected",
                expected,
            )
            break


def cmd_test(exp):
    if exp == "exp_1":
        from exp_1 import Algorithm
    elif exp == "exp_2":
        from exp_2 import Algorithm
    else:
        raise Exception("Unknown exp: {}".format(exp))

    pb_filepath = os.path.join(exp, "frozen.pb")
    checkpoint_path = os.path.join(exp)

    infer = get_inference(Algorithm())
    infer_base = get_base_inference(pb_filepath, AlgorithmBase())

    run_test(10, infer, infer_base, checkpoint_path)


def usage(argv):
    print("-" * 10, "Usage: python3 {} test <exp_1|exp_2>".format(argv[0]))


def main(argv):
    import tensorflow as tf

    tf.logging.set_verbosity(tf.logging.ERROR)
    if len(argv) < 3:
        usage(argv)
        return

    cmd = argv[1]
    if cmd == "test":
        cmd_test(argv[2])
    else:
        raise Exception("Unknown argv: {}".format(argv))


if __name__ == "__main__":
    import sys

    main(sys.argv)
