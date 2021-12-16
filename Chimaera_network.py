import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Change this to train quickly and debuggear throught the code
debug = False
batch_size = 64






#SET THE CLASSES
classes = ('square', 'triangle', 'circle', 'chimaera')

num_channels = 6

#NEURAL NETWORK
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=13, padding="same")
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=11, padding="same")
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=9, padding="same")
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=7, padding="same")
        self.conv5 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=5, padding="same")
        self.fc1 = nn.Linear(25*25*num_channels, 84)
        self.fc2 = nn.Linear(84, 4)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def run_chimaera(ex_num):
    with open("square.npy", "rb") as f:
        squares = np.load(f)

    with open("triangle.npy", "rb") as f:
        triangles = np.load(f)

    with open("circle.npy", "rb") as f:
        circles = np.load(f)

    with open("chimaeras_v2.npy", "rb") as f:
        chimaeras = np.load(f)

    squares = np.array(squares, dtype=np.float64)
    squares *= 2 / 255
    squares -= 1
    triangles = np.array(triangles, dtype=np.float64)
    triangles *= 2 / 255
    triangles -= 1
    circles = np.array(circles, dtype=np.float64)
    circles *= 2 / 255
    circles -= 1
    chimaeras = chimaeras[:, :, :, 0]
    chimaeras = np.array(chimaeras, dtype=np.float64)
    chimaeras *= 2 / 255
    chimaeras -= 1
    all_data = np.vstack((squares, triangles, circles, chimaeras))
    print(all_data.shape)
    all_data = np.expand_dims(all_data, axis=1)
    print(all_data.shape)
    labels = np.zeros(squares.shape[0])
    labels_all = np.hstack((labels, labels + 1, labels + 2, labels + 3))
    print(labels_all.shape)
    tensor_features = torch.Tensor(all_data)
    tensor_labels = torch.Tensor(labels_all)
    tensor_labels = tensor_labels.long()
    my_dataset = TensorDataset(tensor_features, tensor_labels)
    my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)

    squares = None
    triangles = None
    circles = None
    del squares
    del triangles
    del circles

    # LOAD TEST DATA
    with open("test_square.npy", "rb") as f:
        test_squares = np.load(f)

    with open("test_triangle.npy", "rb") as f:
        test_triangles = np.load(f)

    with open("test_circle.npy", "rb") as f:
        test_circles = np.load(f)

    with open("test_chimaeras_v2.npy", "rb") as f:
        test_chimaeras = np.load(f)

    test_squares = np.array(test_squares, dtype=np.float64)
    test_squares *= 2 / 255
    test_squares -= 1
    test_triangles = np.array(test_triangles, dtype=np.float64)
    test_triangles *= 2 / 255
    test_triangles -= 1
    test_circles = np.array(test_circles, dtype=np.float64)
    test_circles *= 2 / 255
    test_circles -= 1
    test_chimaeras = test_chimaeras[:, :, :, 0]
    test_chimaeras = np.array(test_chimaeras, dtype=np.float64)
    test_chimaeras *= 2 / 255
    test_chimaeras -= 1
    # test_chimaeras = np.expand_dims(test_chimaeras, axis=1)

    all_testing_data = np.vstack((test_squares, test_triangles, test_circles, test_chimaeras))
    print(all_testing_data.shape)
    all_testing_data = np.expand_dims(all_testing_data, axis=1)
    print(all_testing_data.shape)
    test_squares = None
    test_triangles = None
    test_circles = None
    test_chimaeras = None
    del test_squares
    del test_triangles
    del test_circles
    del test_chimaeras

    labels = np.zeros(10000)
    labels_all = np.hstack((labels, labels + 1, labels + 2, labels + 3))
    tensor_labels = torch.Tensor(labels_all)
    tensor_labels = tensor_labels.long()
    tensor_features = torch.Tensor(all_testing_data)
    my_test_dataset = TensorDataset(tensor_features, tensor_labels)
    my_test_data = DataLoader(my_test_dataset, batch_size=64, shuffle=False)  # important not suffle the test only train


    #TRAINING
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.002)


    performance_over_time = []
    accuracy_over_time = []
    accuracy_triangle_over_time = []
    accuracy_circle_over_time = []
    accuracy_square_over_time = []
    accuracy_chimaeras_over_time = []
    list_of_lists = [accuracy_square_over_time, accuracy_triangle_over_time, accuracy_circle_over_time, accuracy_chimaeras_over_time]
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(my_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                performance_over_time.append(running_loss/20)
                running_loss = 0.0
                correct = 0
                total = 0
                correct_pred = {classname: 0 for classname in classes}
                total_pred = {classname: 0 for classname in classes}
                n_classes = 4
                confusion_matrix = torch.zeros(n_classes, n_classes)
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    ouput = pd.DataFrame(columns=['shapes', 'labels', 'outputs', 'predictions'])
                    for data in my_test_data:

                        # to calculate accuracy :
                        shapes, labels = data
                        # calculate outputs by running images through the network
                        outputs = net(shapes)
                        # the class with the highest energy is what we choose as prediction
                        _, predictions = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predictions == labels).sum().item()
                        ouput = ouput.append({'shapes': shapes, 'labels': labels, 'outputs': outputs, 'predictions': predictions}, ignore_index=True)
                        for label, prediction in zip(labels, predictions):
                            if label == prediction:
                                correct_pred[classes[label]] += 1
                            total_pred[classes[label]] += 1
                            confusion_matrix[label.view(-1).long(), prediction.view(-1).long()] += 1
                    print(confusion_matrix)

                    print('Accuracy of the network on the 30000 test shapes: %d %%' % (100 * correct / total))
                    accuracy_over_time.append(100*correct/total)
                    for i, (classname, correct_count) in enumerate(correct_pred.items()):
                        accuracy = 100 * float(correct_count) / total_pred[classname]
                        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))
                        list_of_lists[i].append(accuracy)
    ouput.to_csv(f'./chimaera/outputs_of_shapes_exp{ex_num}.csv')
    print('Finished training')


    # save last confusion matrix:
    df_confusion_matrix = pd.DataFrame(confusion_matrix).astype(int)
    df_confusion_matrix.to_csv(f'./chimaera/confusion_matrix_{ex_num}.csv') # remainder: change name after every experiment


    #Save every the data of every experiment
    dict = {'general performance': performance_over_time, 'general accuracy': accuracy_over_time, 'accuracy square': accuracy_square_over_time,
        'accuracy circle': accuracy_circle_over_time, 'accuracy triangle': accuracy_triangle_over_time, 'accuracy_chimaera': accuracy_chimaeras_over_time}
    df = pd.DataFrame(dict)
    df.to_csv(f'./chimaera/info_experiment{ex_num}.csv') # remainder: change name after every experiment

    #create figure performance
    fig0, ax = plt.subplots()
    ax.plot([i for i in range(len(performance_over_time))], performance_over_time)
    ax.set_xlabel('Time')
    ax.set_ylabel('Performance')
    plt.title('Performance over time')
    plt.savefig(f'./chimaera/Performance_of_experiment_{ex_num}.png')  # remainder: change name after every experiment


    #create figure of accuracy
    fig1, ax = plt.subplots()
    ax.plot([i for i in range(len(accuracy_over_time))], accuracy_over_time )
    ax.set_xlabel('Time')
    ax.set_ylabel('Accuracy')
    plt.title('Accuracy over time')
    plt.savefig(f'./chimaera/Accuracy_of_experiment_{ex_num}.png')  # remainder: change name after every experiment

    #create figure accuracy over time for the different classes
    fig2, ax = plt.subplots()
    ax.plot([i for i in range(len(accuracy_over_time))], accuracy_circle_over_time)
    ax.plot([i for i in range(len(accuracy_over_time))], accuracy_triangle_over_time)
    ax.plot([i for i in range(len(accuracy_over_time))], accuracy_square_over_time)
    ax.plot([i for i in range(len(accuracy_over_time))], accuracy_chimaeras_over_time)
    ax.legend(['Circle', 'Triangle', 'Square', 'Chimaera'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Accuracy')
    plt.title('Compared accuracy of different shapes over time')
    plt.savefig(f'./chimaera/Compared_accuracy_of_experiment_{ex_num}.png') # remainder: change name after every experiment


    # save our trained model:
    PATH = f'./chimaera/chimaera_network_{ex_num}.pth'
    torch.save(net.state_dict(), PATH)

# for loading our trained model:
# net = Net()
# net.load_state_dict(torch.load(PATH))


