import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # Const node types
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly

    # The following print prints the nodes, not their value
    print(node1, node2)

    # To print the value we need to run the computational graph
    sess = tf.Session()
    print(sess.run([node1, node2]))


if __name__ == "__main__":
    main()
