import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import datetime
from scipy.misc import imread
from caffe_classes import class_names

# Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

#
# overview dataset
#
# Number of training examples
n_train = np.shape(y_train)[0]

# Number of validation examples
n_validation = np.shape(y_valid)[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
nb_classes = np.unique(y_train).size


print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", nb_classes)

#
# Normalize
#
def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    #min_target = -1.0
    min_target = 0.0
    max_target = 1.0
    min = 0
    max = 255
    return min_target + ( ( (image_data - min)*(max_target - min_target) )/( max - min ) )

X_train = normalize(X_train)
X_valid = normalize(X_valid)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# SOLUTION: Layer 8: Fully Connected. Input = 4096. Output = nb_classes(43).
fc8W  = tf.Variable(tf.truncated_normal(shape=(shape[0], shape[1]), mean = 0, stddev = 0.1))
fc8b  = tf.Variable(tf.zeros(nb_classes))

logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)

EPOCHS = 10
BATCH_SIZE = 128

# Training pipeline
rate = 0.001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Evaluate
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    chart_training_accuracy = []
    chart_validation_accuracy = []

    print("Training...")
    print()
    for i in range(EPOCHS):
        start_time = datetime.datetime.now()
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, int(num_examples / 100), BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        elapsed_time = datetime.datetime.now() - start_time

        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} in {} seconds ...".format(i + 1, elapsed_time.seconds))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

        chart_training_accuracy.append(training_accuracy)
        chart_validation_accuracy.append(validation_accuracy)

    saver.save(sess, './lenet')
    print("Model saved")

#
# Inference
#
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
