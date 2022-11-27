
RANDOM_STATE = 0
'''
For an accurate comparison of the different methods, one can train a CGAN for a large number of iterations, 
 and save the weights of the generator as the training proceeds. Then all methods can use the same GAN which is 
 saved in different iterations. Step size needs to be the same as epoch unit parameter. If such weights are not available,
 for each method, a new CGAN is built and trained.:'''
RELATIVE_ADDRESS_TO_SAVED_GENERATORS = 'saved_generators/'


"""GAN Class
The following written on top of the code obtained from:
https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py 
"""

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

from sklearn.utils import shuffle
class CGAN():
    def __init__(self, X_train, y_train, number_of_generated_samples_perclass):
        self.X_train = X_train
        self.y_train = y_train 
        self.img_shape = (X_train.shape[1],1)
        self.num_classes = 2
        self.latent_dim = 100
        self.total_trained_epochs_as_of_now = 0
        self.number_of_generated_samples = number_of_generated_samples_perclass


        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128):
        X_train,y_train = self.X_train, self.y_train

        self.total_trained_epochs_as_of_now +=1

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # self.sample_images()

    def sample_images(self):
        r, c = 2, 5
        if r*c != self.num_classes: raise NameError('the number of rows and columns should be equal to the number of classes')
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape(28,28), cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % self.total_trained_epochs_as_of_now)
        plt.close()


    def mass_generator(self):
      data = []
      label = []
      for class_ in range(self.num_classes):
        noise = np.random.normal(0, 1, (self.number_of_generated_samples, self.latent_dim)) # latent_dim = 50

        target_class_fake_label = np.ones(self.number_of_generated_samples).reshape(-1,1) * class_

        target_class_fake_data = self.generator.predict([noise, target_class_fake_label])

        data.append(target_class_fake_data)

        label.append(target_class_fake_label)

      data = np.array(data)
      data_shape = data.shape
      data = data.reshape(-1,data_shape[2],)


      label = np.array(label).reshape(-1,)


      data, label = shuffle(data, label, random_state=2)

      return data, label


"""#Oracle Classes:

## Oracle Base Class
"""

def get_devisors_helper(number):
  divs = []
  for i in range(1, int(number / 2) + 1):
      if number % i == 0:
          divs.append(i)
  divs.append(number)

  return np.array(divs)

def get_closest_two_numbers_to_squareroot_of_given_number_helper(number):

  numbers = get_devisors_helper(number)

  len_numbers = numbers.size
  devide_by_2 = len_numbers%2
  middle = int(len_numbers/2)
  if devide_by_2==0:
    return numbers[middle - 1], numbers[middle]

  return numbers[ middle ], numbers[ middle ]

get_closest_two_numbers_to_squareroot_of_given_number_helper(32*32*3)

from abc import ABC, abstractmethod
from tensorflow.keras.layers import Input, Resizing, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

import numpy as np

class Oracle(ABC):



  def score(self):
    pass

  def neural_net_classifier_builder(self,
                                    data,
                                    label,
                                    number_of_hidden_layers_for_classifier,
                                    number_of_neurons_in_layer_for_classifier, 
                                    number_of_epochs_for_training_classifier):

    number_of_classes = int(np.amax(self.label) + 1) #labels start from 0

    classifier = Sequential()
    classifier.add(Input(shape=(data.shape[-1],) ) )

    for i in range(number_of_hidden_layers_for_classifier):
      classifier.add(Dense(number_of_neurons_in_layer_for_classifier, activation='relu'))

    if number_of_classes == 2: 
      classifier.add(Dense(1, activation='sigmoid'))
      classifier.compile(loss='binary_crossentropy', optimizer='adam')
    else:
      classifier.add(Dense(number_of_classes, activation='softmax'))
      classifier.compile(loss='categorical_crossentropy', optimizer='adam')

    classifier.fit(data, label,
                        epochs=number_of_epochs_for_training_classifier,
                        batch_size=32, verbose=0, validation_split=0.0)
    return classifier

  def altered_inception_feature_extractor(self, input_dim=784, number_of_channels=1):
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(75,75,3))

    twoD_shape = get_closest_two_numbers_to_squareroot_of_given_number_helper( int( input_dim/ number_of_channels) )
    ultimate_shape = (twoD_shape[0], twoD_shape[1],number_of_channels)
    
    input_img = Input(shape=(input_dim, ))
    reshaped_img = Reshape( ultimate_shape )(input_img)
    resized_img = Resizing(75, 75)(reshaped_img)
    concat_img = Concatenate()([resized_img, resized_img, resized_img])  

    output = inception(concat_img)

    altered_inception = Model(inputs=input_img, outputs=output)
    return altered_inception

  def altered_inception_classifier(self, input_shape=(784,)):
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    input_img = Input(shape=input_shape)
    reshaped_img = Reshape( (28, 28, 1) )(input_img)
    resized_img = Resizing(299, 299)(reshaped_img)
    concat_img = Concatenate()([resized_img, resized_img, resized_img])  

    output = inception(concat_img)

    output = Dense(1000, activation='softmax')(output)

    altered_inception = Model(inputs=input_img, outputs=output)
    return altered_inception

"""##Oracle FID Class"""

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm



class OracleFID(Oracle):
  def __init__(self,
               cgan_object):

    self.cgan_object = cgan_object

    self.unlabeled_data = self.cgan_object.X_train

    self.feature_extractor = self.altered_inception_feature_extractor()

    self.metric = 'FID'




    #source: https://machinelearningmastery.com/
  def score(self):
    generated_X_train, _ = self.cgan_object.mass_generator()
    #Randomly select n data from real data; n = number of generated data:
    chosen_X_train_indices = np.random.choice(
        range(self.unlabeled_data.shape[0]),
         size = generated_X_train.shape[0],
         replace = True)
    chosen_X_train = self.unlabeled_data[chosen_X_train_indices]
    # calculate activations
    act1 = self.feature_extractor(generated_X_train).numpy()
    act2 = self.feature_extractor(chosen_X_train).numpy()
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return -abs(fid)

#Testing:
# cgan_obj= CGAN(X_train=X_train,
#                y_train=y_train,
#                number_of_generated_samples_perclass = 500)

# fid_obj = OracleFID(cgan_object = cgan_obj)

# fid_obj.score()

"""##Oracle Modified FID Class"""

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

class OracleFCD(Oracle):
  def __init__(self,
               cgan_object,
               number_of_epochs_for_training_feature_extractor):
    self.number_of_epochs_for_training_feature_extractor = number_of_epochs_for_training_feature_extractor

    self.cgan_object = cgan_object

    self.unlabeled_data = self.cgan_object.X_train

    self.feature_extractor = self.feacture_extractor_helper(self.unlabeled_data)

    self.metric = 'FID'



  #source: https://machinelearningmastery.com/
  def feacture_extractor_helper(self, unlabeled_data):
    X = unlabeled_data
    # number of input columns
    n_inputs = X.shape[1]
    # split into train test sets
    X_train = X#, X_test = train_test_split(X, test_size=0.33, random_state=1)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    #X_test = t.transform(X_test)
    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    n_bottleneck = round(float(n_inputs) / 2.0)
    bottleneck = Dense(n_bottleneck)(e)
    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # plot the autoencoder
    # fit the autoencoder model to reconstruct input
    model.fit(X_train,
                        X_train,
                        epochs=self.number_of_epochs_for_training_feature_extractor,
                        batch_size=16,
                        verbose=0)
    encoder = Model(inputs=visible, outputs=bottleneck)
    return encoder

    #source: https://machinelearningmastery.com/
  def score(self):
    generated_X_train, _ = self.cgan_object.mass_generator()

    # calculate activations
    act1 = self.feature_extractor(generated_X_train).numpy()
    act2 = self.feature_extractor(self.unlabeled_data).numpy()
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return -abs(fid)

#Testing:
# cgan_obj= CGAN(X_train=X_train, 
#                y_train=y_train, 
#                number_of_generated_samples_perclass = 12)

# modified_fid_obj = OracleFCD(cgan_object = cgan_obj,
#                number_of_epochs_for_training_feature_extractor = 0)

# modified_fid_obj.score()

"""## Oracle IS Class"""

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp

class OracleIS(Oracle):
  def __init__(self,
               cgan_object):
    
    self.cgan_object = cgan_object
    self.data = self.cgan_object.X_train
    self.label = self.cgan_object.y_train

    self.classifier = self.altered_inception_classifier()

    self.metric = 'IS'


    #source: https://machinelearningmastery.com/
  def score(self, n_split=10, eps=1E-16):

    generated_X_train, _ = self.cgan_object.mass_generator()

    yhat = self.classifier.predict(generated_X_train)
    scores = list()
    n_part = floor(generated_X_train.shape[0] / n_split)
    for i in range(n_split):
      # retrieve p(y|x)
      ix_start, ix_end = i * n_part, i * n_part + n_part
      p_yx = yhat[ix_start:ix_end]
      # calculate p(y)
      p_y = expand_dims(p_yx.mean(axis=0), 0)
      # calculate KL divergence using log probabilities
      kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
      # sum over classes
      sum_kl_d = kl_d.sum(axis=1)
      # average over images
      avg_kl_d = mean(sum_kl_d)
      # undo the log
      is_score = exp(avg_kl_d)
      # store
      scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg

#Testing:
# cgan_obj= CGAN(X_train=data, 
#                y_train=labels, 
#                number_of_generated_samples_perclass = 500)

# is_obj = OracleIS(cgan_object = cgan_obj)

# cgan_obj.train(1000)

# is_obj.score()

"""## Oracle Modified IS Class"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp

class OracleCDS(Oracle):
  def __init__(self,
               cgan_object,
               number_of_hidden_layers_for_classifier,
               number_of_neurons_in_layer_for_classifier, 
               number_of_epochs_for_training_classifier):
    
    self.cgan_object = cgan_object
    self.data = self.cgan_object.X_train
    self.label = self.cgan_object.y_train

    self.classifier = self.neural_net_classifier_builder(self.data,
                                                         self.label,
                                                         number_of_hidden_layers_for_classifier,
                                                         number_of_neurons_in_layer_for_classifier,
                                                         number_of_epochs_for_training_classifier)
    self.metric = 'IS'


    #source: https://machinelearningmastery.com/
  def score(self, n_split=10, eps=1E-16):

    generated_X_train, _ = self.cgan_object.mass_generator()

    yhat = self.classifier.predict(generated_X_train)
    scores = list()
    n_part = floor(generated_X_train.shape[0] / n_split)
    for i in range(n_split):
      # retrieve p(y|x)
      ix_start, ix_end = i * n_part, i * n_part + n_part
      p_yx = yhat[ix_start:ix_end]
      # calculate p(y)
      p_y = expand_dims(p_yx.mean(axis=0), 0)
      # calculate KL divergence using log probabilities
      kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
      # sum over classes
      sum_kl_d = kl_d.sum(axis=1)
      # average over images
      avg_kl_d = mean(sum_kl_d)
      # undo the log
      is_score = exp(avg_kl_d)
      # store
      scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg

"""## Oracle CAS-syn Class"""

from sklearn.metrics import classification_report
from statistics import mean

class OracleCAS_syn(Oracle):
  def __init__(self,
               cgan_object,
               number_of_hidden_layers_for_classifier,
               number_of_neurons_in_layer_for_classifier, 
               number_of_epochs_for_training_classifier):
    
    
    self.cgan_object = cgan_object
    self.data = self.cgan_object.X_train
    self.label = self.cgan_object.y_train

    self.classifier = self.neural_net_classifier_builder(self.data,
                                                         self.label,
                                                         number_of_hidden_layers_for_classifier,
                                                         number_of_neurons_in_layer_for_classifier,
                                                         number_of_epochs_for_training_classifier)
    self.metric = 'F1-score'



  def score(self):
    generated_data, generated_label = self.cgan_object.mass_generator()

    y_pred = self.classifier.predict(generated_data)

    number_of_classes = int(np.amax(self.label) + 1) #labels start from 0

    if number_of_classes == 2: 
      y_pred[y_pred>= 0.5] = 1.
      y_pred[y_pred< 0.5] = 0.
    else: 
      y_pred = np.argmax(y_pred, axis=1)

    result = classification_report(generated_label, y_pred, output_dict=True)
    f1_scores = []
    for r in range(number_of_classes):
      f1_scores.append(result[str(r)+'.0']['f1-score'])

    f1_scores_np = np.array(f1_scores)

    return np.average(f1_scores_np)

"""## Oracle CAS_real Class"""

from sklearn.metrics import classification_report
from statistics import mean

class OracleCAS_real(Oracle):
  def __init__(self,
               cgan_object,
               number_of_hidden_layers_for_classifier,
               number_of_neurons_in_layer_for_classifier, 
               number_of_epochs_for_training_classifier):
    
    
    self.cgan_object = cgan_object
    self.data = self.cgan_object.X_train
    self.label = self.cgan_object.y_train

    self.number_of_epochs_for_training_classifier = number_of_epochs_for_training_classifier

    #this classifier must be trained with the generated data, thus epochs=0
    self.classifier = self.neural_net_classifier_builder(self.data,
                                                         self.label,
                                                         number_of_hidden_layers_for_classifier = number_of_hidden_layers_for_classifier,
                                                         number_of_neurons_in_layer_for_classifier = number_of_neurons_in_layer_for_classifier,
                                                         number_of_epochs_for_training_classifier = 0)

    self.metric = 'F1-score'

  def score(self):
    generated_data, generated_label = self.cgan_object.mass_generator()
    
    self.classifier.fit(generated_data, generated_label,
                        epochs=self.number_of_epochs_for_training_classifier,
                        batch_size=32, verbose=0, validation_split=0.0)

    y_pred = self.classifier.predict(self.data)

    number_of_classes = int(np.amax(self.label) + 1) #labels start from 0
    if number_of_classes == 2: 
      y_pred[y_pred>= 0.5] = 1.
      y_pred[y_pred< 0.5] = 0.
    else: 
      y_pred = np.argmax(y_pred, axis=1)

    result = classification_report(self.label, y_pred, output_dict=True)
    f1_scores = []
    for r in range(number_of_classes):
      f1_scores.append(result[str(r) + '.0']['f1-score']) #ERROR: 0.0

    f1_scores_np = np.array(f1_scores)

    return np.average(f1_scores_np)

"""#WTST Class

##Class
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import copy
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model


class WTST():

  def __init__(self,
               oracle,
               number_of_accepted_failed_attempts,
               epoch_unit,
               dataset_name ,
               maj_count ,
               im ,
               fold):
    
    self.number_of_accepted_failed_attempts = number_of_accepted_failed_attempts
    self.epoch_unit = epoch_unit

    self.cgan_object = oracle.cgan_object

    self.oracle = oracle


    self.all_scores = []


    self.best_generator_yet = copy.deepcopy(self.cgan_object.generator)



    self.maj_count = maj_count
    self.im = im
    self.fold = fold
    self.dataset_name = dataset_name

  def main(self):

    best_score_yet = -10000000000
    number_of_failed_attempts = 0
    self.best_epoch = 0


    current_epoch = 0

    while number_of_failed_attempts <= self.number_of_accepted_failed_attempts  :
      current_epoch += self.epoch_unit

      try:
          path = RELATIVE_ADDRESS_TO_SAVED_GENERATORS + self.dataset_name  + '/' + str(self.maj_count) +'_'+str(self.im)+'_'+str(self.fold)+'/'+str(current_epoch)
          generator = load_model(path)
          self.cgan_object.generator.set_weights(generator.get_weights())
      except:
        self.cgan_object.train(epochs=self.epoch_unit)

  
      score = self.oracle.score()
      self.all_scores.append(score)
      if score <= best_score_yet:
        number_of_failed_attempts +=1
        #print(' failor at :', current_epoch, ' - Score: ', score)
      else:
        self.best_generator_yet.set_weights(self.cgan_object.generator.get_weights())
        self.best_epoch = current_epoch
        best_score_yet = score
        number_of_failed_attempts = 0
        #print(' SUCCESS at :', current_epoch,' - Score: ', score)

    
                

  def plot(self):
    all_scores_np = np.array(self.all_scores)


    plt.clf()

    x_axis = np.array(list(range(all_scores_np.shape[0]))) * self.epoch_unit

    ax = plt.subplot()

    plt.plot(x_axis,all_scores_np,'g--', label=1)


    plt.ylabel(self.oracle_object.metric)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    ticks = (ax.get_xticks())
    ticks =ticks.astype(int)
    ax.set_xticklabels(ticks)
    plt.rcParams["figure.figsize"] = (20,6)
    plt.legend()
    plt.show()

  def balance(self, generator):
    data = np.copy(self.cgan_object.X_train)
    label = np.copy(self.cgan_object.y_train)

    classes = np.unique(label)
    classes_counts = []
    for c in classes:
      classes_counts.append( np.count_nonzero(label == c) )

    maj = max(classes_counts)

    classes_counts = np.array(classes_counts)

    how_many_samples_to_create_for_each_class = -classes_counts + maj

    for sample_index in range(len(how_many_samples_to_create_for_each_class)):
      how_many_for_this_class = how_many_samples_to_create_for_each_class[sample_index]
      if how_many_for_this_class == 0 : continue

      labels = np.ones(how_many_for_this_class).reshape(-1,1) * sample_index
      noise = np.random.normal(0, 1, (how_many_for_this_class, self.cgan_object.latent_dim))

      gen_imgs = generator.predict([noise, labels])

      data = np.append(data, gen_imgs.reshape(-1, data.shape[1]), axis=0 ) 
      label = np.append(label, labels.reshape(-1,))

    data, label = shuffle(data, label, random_state=RANDOM_STATE)

    return data, label



from sklearn.utils import shuffle

def imbalance(data, label, maj_class_count, min_maj_rate):

  first_class = data[label == 0]
  first_class = shuffle(first_class, random_state=0)
  first_class = first_class[0:maj_class_count]

  second_class = data[label == 1]
  second_class = shuffle(second_class, random_state=0)
  second_class_count = int(maj_class_count*min_maj_rate)
  second_class = second_class[0:second_class_count]

  data = np.concatenate((first_class,second_class))
  label = np.concatenate( (np.zeros(first_class.shape[0]), np.ones(second_class.shape[0]) ) ) 
  
  return shuffle(data, label, random_state=RANDOM_STATE)

"""##Write to CSV Function"""

from csv import writer

def write_to_csv(
    dataset_name,
    oracle_param_oracle_name,
    oracle_param_number_of_epochs_for_training_feature_extractor,
    oracle_param_number_of_hidden_layers_for_classifier,
    oracle_param_number_of_neurons_in_layer_for_classifier, 
    oracle_param_number_of_epochs_for_training_classifier,
    autoGAN_param_number_of_accepted_failed_attempts,
    autoGAN_param_epoch_unit,
    gan_param_number_of_generated_samples_perclass,
    maj_class_count,
    min_maj_rate,
    maj_f1,
    min_f1,
    stopping_epoch,
    fold):
  
  with open('results_'+dataset_name+'.csv', 'a') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow([
                            dataset_name,
                            oracle_param_oracle_name,
                            oracle_param_number_of_epochs_for_training_feature_extractor,
                            oracle_param_number_of_hidden_layers_for_classifier,
                            oracle_param_number_of_neurons_in_layer_for_classifier, 
                            oracle_param_number_of_epochs_for_training_classifier,
                            autoGAN_param_number_of_accepted_failed_attempts,
                            autoGAN_param_epoch_unit,
                            gan_param_number_of_generated_samples_perclass,
                            maj_class_count,
                            min_maj_rate,
                            maj_f1,
                            min_f1,
                            stopping_epoch,
                            fold])
    f_object.close()

write_to_csv(
    dataset_name = 'dataset_name',
    oracle_param_oracle_name = 'oracle_param_oracle_name',
    oracle_param_number_of_epochs_for_training_feature_extractor = 'oracle_param_number_of_epochs_for_training_feature_extractor',
    oracle_param_number_of_hidden_layers_for_classifier = 'oracle_param_number_of_hidden_layers_for_classifier',
    oracle_param_number_of_neurons_in_layer_for_classifier = 'oracle_param_number_of_neurons_in_layer_for_classifier', 
    oracle_param_number_of_epochs_for_training_classifier = 'oracle_param_number_of_epochs_for_training_classifier',
    autoGAN_param_number_of_accepted_failed_attempts = 'autoGAN_param_number_of_accepted_failed_attempts',
    autoGAN_param_epoch_unit= 'autoGAN_param_epoch_unit',
    gan_param_number_of_generated_samples_perclass = 'gan_param_number_of_generated_samples_perclass',
    maj_class_count = 'maj_class_count',
    min_maj_rate = 'min_maj_rate',
    maj_f1 = 'maj_f1',
    min_f1 ='min_f1',
    stopping_epoch = 'stopping_epoch',
    fold = 'fold')

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

def oversample(
              dataset_name,
              data,
              labels,
              autoGAN_param_number_of_accepted_failed_attempts,
              autoGAN_param_epoch_unit,
              gan_param_number_of_generated_samples_perclass,
              classifier,
              CAS_syn_number_of_hidden_layers_for_classifier,
              CAS_syn_number_of_neurons_in_layer_for_classifier,
              CAS_syn_number_of_epochs_for_training_classifier,
               
              CAS_real_number_of_hidden_layers_for_classifier,
              CAS_real_number_of_neurons_in_layer_for_classifier,
              CAS_real_number_of_epochs_for_training_classifier,
               
              CDS_number_of_hidden_layers_for_classifier,
              CDS_number_of_neurons_in_layer_for_classifier,
              CDS_number_of_epochs_for_training_classifier,
               
              oracle_param_number_of_epochs_for_training_feature_extractor,
              no_oracle_training_epochs,
              maj_counts,
              im_ratios,
              oracle_name
              ):

  for maj_count in maj_counts:
    for im in im_ratios:
      print('majority class: ', maj_count, ' -- min to maj ratio: ', im)
      d, l = imbalance(data, labels, maj_count, im)
      fold = 0
      for train_index, test_index in kf.split(d, l):
        fold+=1
        print('fold: ', fold)

        X_train, X_test = d[train_index], d[test_index]
        y_train, y_test = l[train_index], l[test_index]


        cgan_obj= CGAN(X_train=X_train, 
                y_train=y_train, 
                number_of_generated_samples_perclass = gan_param_number_of_generated_samples_perclass)
        


        if oracle_name == 'CAS_syn':
          oracle = OracleCAS_syn(cgan_obj,
                            CAS_syn_number_of_hidden_layers_for_classifier,
                            CAS_syn_number_of_neurons_in_layer_for_classifier,
                            CAS_syn_number_of_epochs_for_training_classifier)

          number_of_hidden_layers_for_classifier = CAS_syn_number_of_hidden_layers_for_classifier
          number_of_neurons_in_layer_for_classifier = CAS_syn_number_of_neurons_in_layer_for_classifier
          number_of_epochs_for_training_classifier = CAS_syn_number_of_epochs_for_training_classifier
          oracle_param_number_of_epochs_for_training_feature_extractor = -1
          
        elif oracle_name == 'CAS_real':
          oracle = OracleCAS_real(cgan_obj,
                            CAS_real_number_of_hidden_layers_for_classifier,
                            CAS_real_number_of_neurons_in_layer_for_classifier,
                            CAS_real_number_of_epochs_for_training_classifier)

          number_of_hidden_layers_for_classifier = CAS_real_number_of_hidden_layers_for_classifier
          number_of_neurons_in_layer_for_classifier = CAS_real_number_of_neurons_in_layer_for_classifier
          number_of_epochs_for_training_classifier = CAS_real_number_of_epochs_for_training_classifier
          oracle_param_number_of_epochs_for_training_feature_extractor = -1


          
        elif oracle_name == 'CDS':
          oracle = OracleCDS(cgan_obj,
                            CDS_number_of_hidden_layers_for_classifier,
                            CDS_number_of_neurons_in_layer_for_classifier,
                            CDS_number_of_epochs_for_training_classifier)

          number_of_hidden_layers_for_classifier = CDS_number_of_hidden_layers_for_classifier
          number_of_neurons_in_layer_for_classifier = CDS_number_of_neurons_in_layer_for_classifier
          number_of_epochs_for_training_classifier = CDS_number_of_epochs_for_training_classifier
          oracle_param_number_of_epochs_for_training_feature_extractor = -1



        elif oracle_name == 'FCD':
          oracle = OracleFCD(cgan_obj,
                            oracle_param_number_of_epochs_for_training_feature_extractor)

          number_of_hidden_layers_for_classifier = -1
          number_of_neurons_in_layer_for_classifier = -1
          number_of_epochs_for_training_classifier = -1
          oracle_param_number_of_epochs_for_training_feature_extractor = oracle_param_number_of_epochs_for_training_feature_extractor
        
        elif oracle_name == 'FID':
          oracle = OracleFID(cgan_obj)
          number_of_hidden_layers_for_classifier = -1
          number_of_neurons_in_layer_for_classifier = -1
          number_of_epochs_for_training_classifier = -1
          oracle_param_number_of_epochs_for_training_feature_extractor = -1
        
          
        elif oracle_name == 'Fixed':
          oracle = -1
          try:
              path = RELATIVE_ADDRESS_TO_SAVED_GENERATORS + dataset_name  + '/' + str(maj_count) +'_'+str(im)+'_'+str(fold)+'/'+str(no_oracle_training_epochs)
              generator = load_model(path)
          except:
              cgan_obj.train(no_oracle_training_epochs)
              generator = cgan_obj.generator

          number_of_hidden_layers_for_classifier = -1
          number_of_neurons_in_layer_for_classifier = -1
          number_of_epochs_for_training_classifier = -1
          oracle_param_number_of_epochs_for_training_feature_extractor = -1
          stopping_epoch = no_oracle_training_epochs

        elif oracle_name == 'Random':
          oracle = -1
          generator = cgan_obj.generator

          number_of_hidden_layers_for_classifier = -1
          number_of_neurons_in_layer_for_classifier = -1
          number_of_epochs_for_training_classifier = -1
          oracle_param_number_of_epochs_for_training_feature_extractor = -1
          stopping_epoch = 0




        #special case for 'Fixed' and 'random':
        if oracle == -1:
          dd, ll = balance_stand_alone(generator,X_train,y_train)
          maj_f1, min_f1 = classifier(dd, ll, X_test, y_test)
        else:
          autoGAN_obj = WTST(oracle,
          number_of_accepted_failed_attempts = autoGAN_param_number_of_accepted_failed_attempts,
          epoch_unit = autoGAN_param_epoch_unit,
          dataset_name = dataset_name,
          maj_count = maj_count,
          im = im,
          fold = fold)
        
          autoGAN_obj.main()

          dd, ll = autoGAN_obj.balance(autoGAN_obj.best_generator_yet)
          maj_f1, min_f1 = classifier(dd, ll, X_test, y_test)
          stopping_epoch = autoGAN_obj.best_epoch





        write_to_csv(
        dataset_name,
        oracle_name,
        oracle_param_number_of_epochs_for_training_feature_extractor,
        number_of_hidden_layers_for_classifier,
        number_of_neurons_in_layer_for_classifier,
        number_of_epochs_for_training_classifier,
        autoGAN_param_number_of_accepted_failed_attempts,
        autoGAN_param_epoch_unit,
        gan_param_number_of_generated_samples_perclass,
        maj_class_count = maj_count,
        min_maj_rate = im,
        maj_f1 = maj_f1,
        min_f1 = min_f1,
        stopping_epoch = stopping_epoch,
        fold = fold)



def no_oversample(
              dataset_name,
              data,
              labels,  
              classifier,
              maj_counts,
              im_ratios
              ):

  for maj_count in maj_counts:
    for im in im_ratios:
      print('majority class: ', maj_count, ' -- min to maj ratio: ', im)
      d, l = imbalance(data, labels, maj_count, im)
      fold = 0
      for train_index, test_index in kf.split(d, l):
        fold+=1
        X_train, X_test = d[train_index], d[test_index]
        y_train, y_test = l[train_index], l[test_index]

        maj_f1, min_f1 = classifier(X_train, y_train, X_test, y_test)


        write_to_csv(
        dataset_name,
        oracle_param_oracle_name= 'Initial',
        oracle_param_number_of_epochs_for_training_feature_extractor = -1,
        oracle_param_number_of_hidden_layers_for_classifier = -1,
        oracle_param_number_of_neurons_in_layer_for_classifier = -1,
        oracle_param_number_of_epochs_for_training_classifier = -1,
        autoGAN_param_number_of_accepted_failed_attempts = -1,
        autoGAN_param_epoch_unit = -1,
        gan_param_number_of_generated_samples_perclass = -1,
        maj_class_count = maj_count,
        min_maj_rate = im,
        maj_f1 = maj_f1,
        min_f1 = min_f1,
        stopping_epoch = -1,
        fold = fold)


import csv


def return_manual_inspection_results_dict(dataset):
    my_dict = dict()
    with open('visual_inspection_results.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == dataset:
                my_dict[row[1] + '_' + row[2] + '_' + row[3]] = int(row[4])

    return my_dict

def oversample_manual(
              dataset_name,
              data,
              labels,
              autoGAN_param_epoch_unit,
              classifier,
              maj_counts,
              im_ratios
              ):


  iterations_dict = return_manual_inspection_results_dict(dataset_name)
  for maj_count in maj_counts:
    for im in im_ratios:
      print('majority class: ', maj_count, ' -- min to maj ratio: ', im)
      d, l = imbalance(data, labels, maj_count, im)
      fold = 0
      for train_index, test_index in kf.split(d, l):
        fold+=1
        #print('fold: ', fold)

        X_train, X_test = d[train_index], d[test_index]
        y_train, y_test = l[train_index], l[test_index]


        path = RELATIVE_ADDRESS_TO_SAVED_GENERATORS + dataset_name + '/' + str(maj_count) +'_'+str(im)+'_'+str(fold)+'/'+str(iterations_dict[str(maj_count)+'_'+str(im)+'_'+str(fold)])

        generator = load_model(path)
        balance_stand_alone
        dd, ll = balance_stand_alone(generator,X_train,y_train)
        maj_f1, min_f1 = classifier(dd, ll, X_test, y_test)

        write_to_csv(
        dataset_name,
        'manual',
        -1,
        -1,
        -1,
        -1,
        -1,
        autoGAN_param_epoch_unit,
        -1,
        maj_class_count = maj_count,
        min_maj_rate = im,
        maj_f1 = maj_f1,
        min_f1 = min_f1,
        stopping_epoch = str(iterations_dict[str(maj_count)+'_'+str(im)+'_'+str(fold)]),
        fold = fold)

def balance_stand_alone(generator, data, label):

    classes = np.unique(label)
    classes_counts = []
    for c in classes:
      classes_counts.append( np.count_nonzero(label == c) )

    maj = max(classes_counts)

    classes_counts = np.array(classes_counts)

    how_many_samples_to_create_for_each_class = -classes_counts + maj

    for sample_index in range(len(how_many_samples_to_create_for_each_class)):
      how_many_for_this_class = how_many_samples_to_create_for_each_class[sample_index]
      if how_many_for_this_class == 0 : continue

      labels = np.ones(how_many_for_this_class).reshape(-1,1) * sample_index
      noise = np.random.normal(0, 1, (how_many_for_this_class, 100))

      gen_imgs = generator.predict([noise, labels])

      data = np.append(data, gen_imgs.reshape(-1, data.shape[1]), axis=0 ) 
      label = np.append(label, labels.reshape(-1,))

    data, label = shuffle(data, label, random_state=RANDOM_STATE)

    return data, label
