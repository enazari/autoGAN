
RANDOM_STATE = 0



"""#Importing needed classes and functions"""

from utils import CGAN, WTST, imbalance, write_to_csv, oversample, no_oversample, oversample_manual

"""##Defining the Classifier """

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import classification_report

def nn(x,y, x_test, y_test):

    model = Sequential()
    model.add(Dense(5, input_shape=(x.shape[-1],), activation='relu'))
    # model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs=20, batch_size=32, verbose=0, validation_split=0.0)

    y_pred = model.predict(x_test)
    y_pred[y_pred>= 0.5] = 1.
    y_pred[y_pred< 0.5] = 0.
    result = classification_report(y_test, y_pred, output_dict=True)

    maj_f1 = result['0.0']['f1-score']
    min_f1 = result['1.0']['f1-score']

    return maj_f1, min_f1

"""##Import Dataset"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, help="The name of the dataset to be recorded in results CSV file ")
parser.add_argument('--oracle_name', type=str, help="The name of the oracle that should be used. It should be one of the"
                                                    " following: FID, FCD, CDS, CAS_syn, CAS_real. There are two special"
                                                    " named Manual and Initial. Manual means the corresponding function "
                                                    "for manual inspection is called(the labels from manual inspection "
                                                    "should be saved in a CSV file named visual_inspection_results.CSV."
                                                    "Initial means not to use any oversampling method")

parser.add_argument('--majority_class', type=int, help="Majority class label")
parser.add_argument('--minority_class', type=int, help="Minority class label")


args = parser.parse_args()

dataset_name = vars(args)['dataset_name']
oracle_name = vars(args)['oracle_name']
class0 = vars(args)['majority_class']
class1 = vars(args)['minority_class']

#example: python main.py --dataset_name='fmnist79' --oracle_name='CDS' --majority_class=7 --minority_class=9

from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(all_data, all_label), (all_test_data, all_test_label) = fashion_mnist.load_data()

all_data = (all_data.astype(np.float32) - 127.5) / 127.5
X_train = all_data.reshape(-1, 784)
y_train = all_label.reshape(-1, 1)

data = X_train[np.logical_or(y_train==class0, y_train==class1).reshape(60000,)]
labels = y_train[np.logical_or(y_train==class0, y_train==class1).reshape(60000,)].reshape(-1,)

labels[labels==class0] = 0
labels[labels==class1] = 1





# Running the tests

if oracle_name != 'Initial' and oracle_name !='Manual':
    oversample(
                  dataset_name = dataset_name,
                  data = data,
                  labels = labels,
                  autoGAN_param_number_of_accepted_failed_attempts = 10,
                  autoGAN_param_epoch_unit = 100,
                  gan_param_number_of_generated_samples_perclass = 500,
                  classifier = nn,
                  CAS_syn_number_of_hidden_layers_for_classifier = 2,
                  CAS_syn_number_of_neurons_in_layer_for_classifier = 100,
                  CAS_syn_number_of_epochs_for_training_classifier = 100,

                  CAS_real_number_of_hidden_layers_for_classifier = 2,
                  CAS_real_number_of_neurons_in_layer_for_classifier = 20,
                  CAS_real_number_of_epochs_for_training_classifier = 20,

                  CDS_number_of_hidden_layers_for_classifier = 2,
                  CDS_number_of_neurons_in_layer_for_classifier = 100,
                  CDS_number_of_epochs_for_training_classifier = 100,

                  oracle_param_number_of_epochs_for_training_feature_extractor = 200,
                  no_oracle_training_epochs = 9000,
                  maj_counts = [100, 200, 500, 1000],
                  im_ratios = [0.1, 0.2, 0.3, 0.4],
                  oracle_name = oracle_name
                  )

if oracle_name == 'Initial':
    no_oversample(
              dataset_name = dataset_name,
              data = data,
              labels = labels,
              classifier = nn,
              maj_counts=[100, 200, 500, 1000],
              im_ratios=[0.1, 0.2, 0.3, 0.4]
    )

if oracle_name == 'Manual':
    oversample_manual(
              dataset_name = dataset_name,
              data = data,
              labels = labels,
              autoGAN_param_epoch_unit = 100,
              classifier = nn,
              maj_counts = [100, 200, 500, 1000],
              im_ratios = [0.1, 0.2, 0.3, 0.4]
              )

