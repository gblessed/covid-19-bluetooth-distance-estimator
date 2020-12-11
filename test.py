import json
import numpy as np
from tensorflow import keras as ks
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow import math
import tensorflow as tf

from sklearn import svm

def recall_m(y_true, y_pred):
    #tf.print(y_true)
    #y_pred = y_pred.round().astype(int).argmax(axis=1)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    #y_pred = y_pred.round().astype(int).argmax(axis=1)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    y_pred = y_pred.round().astype(int).argmax(axis=1)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


with open(r"c:/Users/HP/Documents/python programs/dataset.json") as f:
    dataset = json.load(f)
training_vals = list()
training_labels  = list()

for dataset_keys in dataset.keys():
    sample_point = dataset[dataset_keys]
    if len(sample_point["distance"])>0:
        cpu_vals = (np.array(dataset[dataset_keys]["cpu_usage"])/ 100.0).mean()
        orientation = np.array(dataset[dataset_keys]["orienation"])
        if (sample_point["os_version"] == "android7.1"):
            osVal = 7
        if (sample_point["UE_model"] == "itel it1508"):
            ueVal = 2
        if (sample_point["UE_model"] == "TECNO F1"):
            ueVal = 1
        if (sample_point["UE_model"] == "Nokia 4.2"):
            ueVal = 3
        
        sample_values = np.array(list((cpu_vals, orientation[0], orientation[1], orientation[2],
                            np.float(sample_point["interference_signals"]), np.float(sample_point["interference_sum"]), osVal, ueVal,
                            sample_point["battery_status"], np.float(sample_point["battery_temp"]), sample_point["I/O"], np.float(sample_point["rssi"]))))
        sample_values.reshape(12,1)
        training_vals.append(sample_values)
        training_labels.append(int(sample_point["distance"])-1)


training_vals = np.array(training_vals)
training_labels = np.array(training_labels)

#print('1length vals is '+str(len(training_vals)))

print(training_vals[0].shape)
input_data_shape = training_vals[0].shape
hidden_activation_function = 'relu'
output_activation_function = 'softmax'

# shuffle data
id = np.random.permutation(len(training_vals))
training_vals, training_labels = training_vals[id], training_labels[id]
print(training_labels)
#print('1length labels is '+str(len(training_labels)))


training_vals, test_vals, training_labels, test_labels = train_test_split(training_vals, training_labels, test_size =0.2)
print('length vals is '+str(len(training_vals)))
print('length labels is '+str(len(training_labels)))
print('length test vals is '+str(len(test_vals)))
print('length test labels is '+str(len(test_labels)))


print('Training Vals Dataset Shape: {}'.format(training_vals.shape))
print('No. of Training Vals Dataset Labels: {}'.format(len(training_labels)))

print('Test Vals Dataset Shape: {}'.format(test_vals.shape))
print('No. of Test Vals Dataset Labels: {}'.format(len(test_labels)))

nn_model = ks.models.Sequential()
nn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='Input_layer'))
nn_model.add(ks.layers.Dense(9, activation=hidden_activation_function, name='Hidden_layer'))
nn_model.add(ks.layers.Dense(5, activation=output_activation_function, name='Output_layer'))

optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy', ks.metrics.Recall(), ks.metrics.Precision()]
nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
#nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)

nn_model.fit(training_vals, training_labels, epochs=50)



#training_loss, training_accuracy,  training_f1score, training_precision, training_recall = nn_model.evaluate(training_vals, training_labels)
training_loss, training_accuracy = nn_model.evaluate(training_vals, training_labels)



print('Training Data Accuracy using NN {}'.format(round(float(training_accuracy),2)))
#print('Training Data Recall using snn {}'.format(round(float(training_recall),2)))
#print('Training Data Precision using snn {}'.format(round(float(training_precision),2)))

y_pred = nn_model.predict(test_vals)
#print("Recall:",metrics.recall_score(test_labels, y_pred, average = 'samples'))

#print(y_pred.round().astype(int).argmax(axis=1))
#print(test_labels)
#print(test_labels)
print ('Results using Neural Network at 50 epochs')
print('Confusion Matrix')
print(metrics.confusion_matrix(test_labels, y_pred.round().astype(int).argmax(axis=1)))

print("Accuracy:",metrics.accuracy_score(test_labels, y_pred.round().astype(int).argmax(axis=1)))

print("Precision:",metrics.precision_score(test_labels, y_pred.round().astype(int).argmax(axis=1), average = 'macro'))

print("Recall:",metrics.recall_score(test_labels,y_pred.round().astype(int).argmax(axis=1), average = 'macro'))

#test_loss, test_accuracy,  recall, precision_m  = nn_model.evaluate(test_vals, test_labels)

#print('Test Data Accuracy using snn {}'.format(round(float(test_accuracy),2)))
#print('Test Data Recall using snn {}'.format(round(float(precision_m),2)))
#print('Test Data Precision using snn {}'.format(round(float(precision_m),2)))






#Create a svm Classifier
clf = svm.SVC(kernel='poly') # Linear Kernel
#print(len(x_train), len(training_labels))
#Train the model using the training sets
clf.fit(training_vals, training_labels)


y_pred = clf.predict(test_vals)
print(y_pred)
print("Results using SVM with polynomial Kernel")
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(test_labels, y_pred, average = 'micro'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(test_labels, y_pred, average = 'micro'))
print(metrics.confusion_matrix(test_labels, y_pred))
#tn, fp, fn, tp = a[0], a[1], b[0], b[1]
