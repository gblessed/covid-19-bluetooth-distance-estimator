# covid-19 bluetooth-distance-estimator
The dataset.json is the dataset we collected for distance estimation. We have collected 272 samples The following are the parameters collected:
- ground distance (label) in meters
- bluetooth rssi
- battery status charging/not charging
- battery temperature
- Count of number of wifi interference signals as seen by UE-1
- sum of wifi rssi values of rssi
- os version
- UE model
- past 10 CPU usage values in % observed during the measurement (with a periodicity of 4s). 
- ambient light intensity level (can be used for detecting indoor/outdoor, levels 0,1,2,3,4,5)
- orientation of device (pitch, azimuth and roll)
# Data collection mechanism
The data collected is using an android app written with Qt Creator using C++ and Java. Only one UE(UE-1) has the Data Collection App. 
The data is collected in concentric circles. UE-1 is  moved along the circumference, while the other UE's are placed in the center.
The PTA observe the rssi as seen by the "UE-1" with reference to the "circumference-UE". (We dont pair the UEs).
orientation of device is noted by flipping the device over in a few directions and checking the corresponding rssi. While indoors we equally collect data when UE-1 is charging.
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

**test-1 : using Neural network**
-Accuracy: **0.45**
-Precision: **0.38**
-Recall: **0.32**

**test-2 using SVM  SVM with polynomial Kernel**
-Accuracy: **0.54**
-Precision: **0.54**
-Recall: **0.54**
