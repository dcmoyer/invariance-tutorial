
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep

IMG_DIM=28
NUM_LABELS=10

def get_data():
  
  #TODO MOVE TO PREPROCESSING
  (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
  
  train_x = train_x.astype(np.float32)
  test_x = test_x.astype(np.float32)
  
  train_x = train_x.reshape( (train_x.shape[0], IMG_DIM**2) ) / 255.0
  test_x = test_x.reshape( (test_x.shape[0], IMG_DIM**2) ) / 255.0
  
  def one_hot(labels):
    num_labels_data = labels.shape[0]
    one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
    one_hot_encoding[np.arange(num_labels_data),labels] = 1
    one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding
  
  train_y = one_hot(train_y)
  test_y = one_hot(test_y)
  
  train_y = train_y.astype(np.float32)
  test_y = test_y.astype(np.float32)
  
  #def standard_scale(X_train, X_test):
  #    preprocessor = prep.StandardScaler().fit(X_train)
  #    X_train = preprocessor.transform(X_train)
  #    X_test = preprocessor.transform(X_test)
  #    return X_train, X_test
  
  #train_x, test_x = standard_scale(train_x, test_x)
  #train_x = train_x / 255.0
  #test_x = test_x / 255.0

  return (train_x, train_y), (test_x, test_y)

