import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # Constant type nodes
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly

    # Operation node types
    node3 = tf.add(node1, node2)

    sess = tf.Session()

    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))


if __name__ == "__main__":
    main()
