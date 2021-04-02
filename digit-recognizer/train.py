import os
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import sys
# sys.path.append("C:\\Users\\faceRecognition\\kaggle\\digit-recognizer")
from net import LeNet5

train_pd = pd.read_csv("C:\\Users\\faceRecognition\\kaggle\\digit-recognizer\\data\\train.csv")
test_pd = pd.read_csv("C:\\Users\\faceRecognition\\kaggle\\digit-recognizer\\data\\test.csv")
print(train_pd.shape)
print(test_pd.shape)

# for i in (10, 25, 1000, 2500):
#     sample_data = data.iloc[i][1:]
#     sample_data = sample_data.values.reshape(28,28)
#     plt.imshow(sample_data)
#     plt.axis("off")
#     plt.show()

# for i in (10, 25, 1000, 2500):
#     print(data.iloc[i][0])

# data.isnull().any().describe()

# import seaborn as sns

# plt.ioff()
# sns.set_theme(style="darkgrid")
# ax = sns.countplot(data=data, x="label")
# plt.show()

# transform input from dataframe to numpy

train_Y = train_pd.label.values
train_X = train_pd.loc[:, train_pd.columns != "label"].values/255
# print(train_Y.shape)
# print(train_X.shape)

# split into train and validation

from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
# print(train_x.shape)
# print(train_y.shape)
print(train_y)

featureTrain = torch.from_numpy(train_x)
targetTrain = torch.from_numpy(train_y).type(torch.LongTensor)
featureTest = torch.from_numpy(valid_x)
targetTest = torch.from_numpy(valid_y).type(torch.LongTensor)

batch_size = 100
num_iters = 2500
num_epoch = num_iters / (len(featureTrain) / batch_size)
num_epoch = int(num_epoch)

train = torch.utils.data.TensorDataset(featureTrain, targetTrain)
test = torch.utils.data.TensorDataset(featureTest, targetTest)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

model = LeNet5()

L = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epoch):
    for i, (image, label) in enumerate(train_loader):
        train = Variable(image.view(100,1,28,28))
        labels = Variable(label)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = L(outputs, label)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(100,1,28,28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))