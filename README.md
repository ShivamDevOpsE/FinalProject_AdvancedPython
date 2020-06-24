# FinalProject_AdvancedPython
Submission of Final Project: German Traffic Signs Detection using Keras and tkinter.

For this project i am using the data available on Kaggle uploaded by German Traffic Signs Recognition Benchmark.
Available on: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

I have used libraries like:
- pandas
- numpy
- keras
- tkinter
- and matplotlib.pyplot

In this project, first we manupulate the data that we have downloaded and save it into numpy array and then train it using typical elements for image classification as X_train, X_test,
and y_train and y_test.
To build and train our neural network i have used 4 convolutional layer along with couple of dense layers. I also included the dropout layer to reduce overfitting in the model.
After the training of the model, you will find the graphs between our accuracy and loss function using matplot library. There are two graphs, the first one is between accuracy of the training set vs accuracy of the validation dataset
and the next graph is between loss of the training set and the loss of the validation data set.
After this, this program will calculate the accuracy_score as sklearn function, basic idea behind it is to calculate the rate the between the predicted labels and the actual labels.

Now after successful training of the model and getting around 90% accuracy in 12 iterations(epochs), we can proceed further to build UI using tkinter.
For this i have defined a dictionary named as classes to describe the sign labels
In our User interface, there will be a main window which consist two buttons, one is to upload the image which we want to classfy and other which actually process that image and shows the result.
Basic dimensions and other property of this user interface you can find in the code.
