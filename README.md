# covid-19 bluetooth-distance-estimator
The dataset.json is the dataset we collected for distance estimation. We have collected 272 samples The following are the parameters collected:
- ground distance (label) in meters
- bluetooth rssi
- battery status charging/not charging
- battery temperature
- number of wifi interfence signals
- sum of wifi rssi values of rssi
- os version
- UE model
- cpu usage  in percentage
- Indoor/Outdoor based on ambient light intensity 
- orientation of device (pitch, azimuth and roll)

# Tranining
The test.py is used to train the data using a neural network and an SVM model. 
It is carried out in python 3.7. The modules and version used are:
tensorflow 2.0.0
sklearn 
numpy
The dataset.json is read and pre-processed and used for training all in test.py before been fed.
An svm with a polynomial kernel is used.
The neural network is setup with 3 layers (9 hidden nodes and 5 output nodes).we use the adam optimizer and sparse_categorical_crossentropy  loss_function. 
We run it for 50 epochs.

# Benchmark Results
We use the 80-20 train test split
Neural network
-Accuracy: 0.45
-Precision: 0.38
-Recall: 0.32
Results using SVM with polynomial Kernel
-Accuracy: 0.54
-Precision: 0.54
-Recall: 0.54
