import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # Placeholders are the way to parametrize the inputs of a node
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    node = a + b  # same as tf.add(a, b)

    sess = tf.Session()

    print(sess.run(
        node,
        {a: 3, b: 4.5}
    ))
    print(sess.run(
        node,
        {a: [1, 3], b: [2, 4]}
    ))


if __name__ == "__main__":
    main()
