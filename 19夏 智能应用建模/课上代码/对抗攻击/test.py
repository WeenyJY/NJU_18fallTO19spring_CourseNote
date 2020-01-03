import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
from tensorflow import keras

# load cifar10 data
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images=train_images.reshape(-1,28,28,1)/255.0
test_images=test_images.reshape(-1,28,28,1)/255.0

# build model 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
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

epsilons = [0.3,0.4,0.5,0.6]#,0.6,0.7,0.8]
alpha=[0.05,0.05,0.05,0.05]#,0.05,0.05,0.05]

def ifgsm(eps,alpha,image,n):
    if n==0:
        return image    
    image_probs = model.predict(image, steps=1)
    perturbations = create_adversarial_pattern(image, image_probs)
    adv_x = image + alpha*perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    return ifgsm(eps,alpha,adv_x,n-1)

for (eps,a) in zip(epsilons,alpha):
    test=test_images
    for i in range(test.shape[0]):
        image = tf.cast(test[i], tf.float32)
        image = tf.image.resize(image, (28,28))
        image = tf.reshape(image, [28,28,1])
        image = image[None, ...]      #tensor, 1,28,28,1
        adv_x=ifgsm(eps,a,image,eps//a)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            array=adv_x.eval()
        test[i]=array[0]
    test_acc=model.evaluate(test, test_labels)[1]
    print('eps = {}, Test accuracy={}'.format(eps,  test_acc))
