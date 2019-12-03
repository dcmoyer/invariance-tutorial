
from data_tools import mnist_dataset
import numpy as np

params = {
  "beta" : 0.1,
  "lambda" : 1.0,
}

(train_x, train_y), (test_x, test_y) = mnist_dataset.get_data()

##
## let's build our VAE network
##

import keras
# sorry, for portability should be just
import keras.backend as K
# but both kl_tools and echo lock us into tensorflow sooo...
import tensorflow as tf

import kl_tools

DIM_Z = 2
DIM_C = mnist_dataset.NUM_LABELS
INPUT_SHAPE=mnist_dataset.IMG_DIM ** 2
ACTIVATION="tanh"

input_x = keras.layers.Input( shape = [INPUT_SHAPE], name="x" )

enc_hidden_1 = keras.layers.Dense(512, activation=ACTIVATION, name="enc_h1")(input_x)
enc_hidden_2 = keras.layers.Dense(512, activation=ACTIVATION, name="enc_h2")(enc_hidden_1)

#stolen straight from the docs
#https://keras.io/examples/variational_autoencoder/
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


z_mean = keras.layers.Dense(DIM_Z, activation="tanh")(enc_hidden_2)
z_log_sigma_sq = keras.layers.Dense(DIM_Z, activation="linear")(enc_hidden_2)

z = keras.layers.Lambda(sampling, output_shape=(DIM_Z,), name='z')([z_mean, z_log_sigma_sq])

## this is the concat operation!
input_c = keras.layers.Input( shape = [DIM_C], name="c")
z_with_c = keras.layers.concatenate([z,input_c])
z_mean_with_c = keras.layers.concatenate([z_mean,input_c])

dec_h1 = keras.layers.Dense(512, activation=ACTIVATION, name="dec_h1")
dec_h2 = keras.layers.Dense(512, activation=ACTIVATION, name="dec_h2")
output_layer = keras.layers.Dense( INPUT_SHAPE, name="x_hat" )

dec_hidden_1 = dec_h1(z_with_c)
dec_hidden_2 = dec_h2(dec_hidden_1)
x_hat = output_layer(dec_hidden_2)

cvae = keras.models.Model(inputs=[input_x,input_c], outputs=x_hat, name="ICVAE")

print(cvae.summary())

##
## make a mean model for outputs
##

mean_dec_hidden_1 = dec_h1(z_mean_with_c)
mean_dec_hidden_2 = dec_h2(mean_dec_hidden_1)
mean_x_hat = output_layer(mean_dec_hidden_2)

mean_cvae = keras.models.Model(
  inputs=[input_x, input_c],
  outputs=mean_x_hat,name="mean_VAE",
)


##
## okay, now we have a network. Let's build the losses
##

recon_loss = keras.losses.mse(input_x, x_hat)
recon_loss *= INPUT_SHAPE #optional, in the tutorial code though

kl_loss = 1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

kl_qzx_qz_loss = kl_tools.kl_conditional_and_marg(z_mean, z_log_sigma_sq, DIM_Z)

#optional add beta param here
# and cite Higgins et al.
cvae_loss = K.mean((1 + params["lambda"]) * recon_loss + params["beta"]*kl_loss + params["lambda"]*kl_qzx_qz_loss)

cvae.add_loss(cvae_loss)

##
##
##

learning_rate = 0.0005
opt = keras.optimizers.Adam(lr=learning_rate)

cvae.compile( optimizer=opt, )

#training?
import os
if not os.path.exists("mnist_icvae.h5"):
  cvae.fit(
    { "x" : train_x, "c" : train_y }, epochs=100
  )
  cvae.save_weights("mnist_icvae.h5")
else:
  cvae.load_weights("mnist_icvae.h5")

exit(1)

n_plot_samps = 10
test_x_hat = mean_cvae.predict(
 { "x" : test_x[:n_plot_samps], "c" : test_y[:n_plot_samps] }
)

##
## plot first N
##

from plot_tools import pics_tools as pic
import matplotlib.pyplot as plt

fig = pic.plot_image_grid( \
  np.concatenate([test_x[:n_plot_samps],test_x_hat], axis=0),
  [mnist_dataset.IMG_DIM, mnist_dataset.IMG_DIM], \
  (2,n_plot_samps) \
)
plt.show()

X_test_set = []
Y_test_set = []

for i in range(n_plot_samps):
  tmp_tile_array = np.tile(test_x[i],[10,1])
  X_test_set.append(test_x[i:(i+1),:])
  X_test_set.append(tmp_tile_array)

  Y_test_set.append(test_y[i:(i+1),:])
  #Y_test_set.append(np.array([[0],[1]]))
  Y_test_set.append(np.eye(10))

X_test_set = np.concatenate(X_test_set, axis=0)
Y_test_set = np.concatenate(Y_test_set, axis=0)

X_test_hat = mean_cvae.predict(
 { "x" : X_test_set, "c" : Y_test_set }
)

plot_collection = []
for i in range(n_plot_samps):
  plot_collection.append( test_x[i:(i+1),:] )
  plot_collection.append( X_test_hat[i*11:(i+1)*11,:] )

plot_collection = np.concatenate( plot_collection, axis=0 )

fig = pic.plot_image_grid( \
  plot_collection,
  [mnist_dataset.IMG_DIM, mnist_dataset.IMG_DIM], \
  (n_plot_samps,12) \
)
plt.show()







