from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, save_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import tensorflow as tf
import cv2


tf.compat.v1.disable_eager_execution()


class Evaluator(object):
    # Evaluator returns the loss and the gradient in two separate functions,
    # but the calculation of the two variables are dependent. This reduces
    # the computation time, since otherwise it would be calculated separately.

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))

    # Remove zero-center by mean pixel. Necessary when working with VGG model
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Format BGR->RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(
        K.permute_dimensions(x, (2, 0, 1)))  # channel first
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    # x_dimension
    a = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    # y_dimension
    b = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def eval_loss_and_grads(x):
    # Evaluate the loss and the gradients respect to the generated image. It
    # is called in the Evaluator, necessary to compute the gradients and the
    # loss as two different functions (limitation of the L-BFGS algorithm)
    # without excessive losses in performance
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


img_nrows = 224
img_ncols = 224
base_image_path = 'images/origin_2.jpg'
style_reference_image_path = 'styles/style_1.jpg'
content_weight = 0.025
style_weight = 1
total_variation_weight = 1
iterations = 10


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(
    preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

model.summary()

loss = K.variable(0.0)

# contribution of content_loss
feature_layers_content = outputs_dict['block5_conv2']
combination_features = feature_layers_content[2, :, :, :]
loss = loss + content_weight * content_loss(
    feature_layers_content,
    combination_features)

# contribution of style_loss
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    # input.shape = [base, style, combination]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    # mean of all output extract
    loss = loss + (style_weight / len(feature_layers)) * sl


# contribution of variation_loss
loss += total_variation_weight * total_variation_loss(combination_image)

# Get the gradients of the generated image
grads = K.gradients(loss, combination_image)
outputs = [loss]
outputs += grads

f_outputs = K.function([combination_image], outputs)

evaluator = Evaluator()

x = preprocess_image(base_image_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = './result/res_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    print('Show image')
    show_img = cv2.imread(fname)

cv2.imshow('Result', show_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
