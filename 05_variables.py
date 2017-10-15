import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # Variables are used to add trainable parameters to the graph
    # The point is to modify the graph to get new outputs with the same input
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    sess = tf.Session()

    # Variables must be always initialized
    init = tf.global_variables_initializer()
    sess.run(init)  # Run the initialization

    print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


if __name__ == "__main__":
    main()
