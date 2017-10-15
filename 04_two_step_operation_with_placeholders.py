import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # Placeholders are the way to parametrize the inputs of a node
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)

    adder_node = a + b  # same as tf.add(a, b)
    add_and_triple = adder_node * 3

    sess = tf.Session()

    print(sess.run(
        add_and_triple,
        {a: 3, b: 4.5}
    ))


if __name__ == "__main__":
    main()
