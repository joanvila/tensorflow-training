import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The images are flattened into a 784-dimensional vector
# None means that a dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])

# We want to multiply the 784-dimensional image vectors by it to produce
# 10-dimensional vectors of evidence for the different numbers (0-9)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# The loss will be calculated with the cross-entropy
# Let's define a placeholder to input the correct answer
y_ = tf.placeholder(tf.float32, [None, 10])

# Implement the cross-entropy function
# A more stable way to do so is with tf.nn.softmax_corss_entropy_with_logits
# on the model
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Apply the optimization algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch the model in an InteractiveSession
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train our model in random batches to make it less expensive
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Check how well trained is the model comparing the prediction (y)
# with the correct value (_y)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# From the list of booleans generated, check the amount of them that are True
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Check the accuracy
print(sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels
}))
