import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.applications.imagenet_utils import decode_predictions


# load cifar10 data
cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


# build model 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#pretrained_model = tf.keras.applications.MobileNet(include_top=True,
#                                                    weights='imagenet')
#pretrained_model.trainable = False

# ImageNet labels


def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]


# load a test image
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = image/255.0
  image = tf.image.resize(image, (32,32))
  image = tf.reshape(image, [32,32,3])
  image = image[None, ...]
  return image

image_path = r'C://Users//Jaqen//Desktop//airplane1.png'
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_png(image_raw)

image = preprocess(image)
image_probs = model.predict(image, steps=1)
y_class = image_probs.argmax(axis=-1)
print('original prediction = {}'.format(y_class))

loss_object = tf.keras.losses.CategoricalCrossentropy()


# generate adversarial noise
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, prediction)
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

perturbations = create_adversarial_pattern(image, image_probs)

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]


for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, 0, 1)
  adv_pred = model.predict(adv_x, steps=1)
  adv_class = adv_pred.argmax(axis=-1)
  print('eps = {}, adv class = {}'.format(eps, adv_class))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.figure()
    plt.imshow(adv_x[0].eval())
    plt.show()

  
  
