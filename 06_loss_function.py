import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # The lost function calculates how far the current model is
    # from the provided data
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # y is the placeholder of the desired values we expect from the regression
    # linear_model - y is a vector which is element is the error delta
    # square de error deltas is a typical loss model for linear regressions
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)

    # We sum all the squared delta errors to create a single scalar
    # for representing in an abstract way the error of all the examples
    loss = tf.reduce_sum(squared_deltas)

    sess = tf.Session()

    # Variables must be always initialized
    init = tf.global_variables_initializer()
    sess.run(init)  # Run the initialization

    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # By reassigning the W and b values to -1. and 1. the result would
    # match the y vector so the final loss would be 0.0. This means
    # we would have guessed the perfect values for W and b.
    # However, the whole point of machine learning is to find those
    # values automatically


if __name__ == "__main__":
    main()
