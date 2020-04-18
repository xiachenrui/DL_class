import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import time
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
# from sklearn import cross_validation

seed = 520
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


path = r"D:\tecent cloud\programs\DL_class\1\data"
train_data = load_mnist(path, kind="train")[0]
train_label = load_mnist(path,kind="train")[1]
test_data = load_mnist(path, kind="t10k")[0]
test_label = load_mnist(path,kind="t10k")[1]


# 自定义数据集
class my_mnist(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


train_dataset = my_mnist(train_data, train_label)
test_dataset = my_mnist(test_data, test_label)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    """定义网络，继承torch.nn.Module"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
cirterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def train(epochs=5):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        best_train_acc = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            inputs = inputs.float()
            labels = labels.long()
            outputs = net(inputs)
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_loss += loss.data[0]
            running_loss += loss.item()
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += float((train_predicted == labels.data).sum())
            train_total += labels.size(0)
            # print(i)
            if i % 200 == 199:
                print('epoch %d [%5d] loss: %.5f acc:%.3f'
                      % (epoch + 1, i + 1, running_loss / train_total, 100 * float(train_correct / train_total)))

        print('Summary of epoch: train %d epoch loss: %.5f  acc: %.3f '
            % (epoch + 1, running_loss / train_total, 100 * float(train_correct / train_total)))
        acc = 100 * float(train_correct / train_total)
        best_train_acc = max(acc, best_train_acc)
        if acc > best_train_acc:
            torch.save(net, 'best_train_model.pt')


def test():
    net.eval()

    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0
    all_labels = np.empty((0, 10))
    all_outputs = np.empty((0, 10))

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.float()
        labels = labels.long()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # 采用one-hot编码
        one_hot_label = label_binarize(labels.numpy(), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        all_labels = np.append(all_labels, one_hot_label, axis=0)
        all_outputs = np.append(all_outputs, outputs.detach().numpy(), axis=0)
        # 在空数组中应用vstack会报错
        # all_labels = np.vstack((all_labels, labels.numpy()))  # 将原有的标签记录到all_labels中，画ROC
        # all_outputs = np.vstack((all_outputs, outputs.detach().numpy()))  # 将预测的得分记录到all_outputs中，画ROC

        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += float((predicted == labels.data).sum())
        acc = float(100 * correct / test_total)

    print('test  loss: %.5f  acc: %.3f ' % (test_loss / test_total, acc))
    ROC_nClass(all_labels, all_outputs, 10)



def show_sample(data, label):
    '''for visualize images '''
    # 将子图组合成一个大图
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True)

    ax = ax.flatten()
    for i in range(10):
        # 布尔型代表是否选取这个维度，后面代表对这个维度进行的操作
        img = train_data[train_label == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    # 去除图中的坐标轴
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def ROC_2class(y_label, y_score):
    '''
    :param y_label: one-hot 编码的测试集正确标签
    :param y_score: 测试样本每一类的得分
    :return: None
    '''
    # Compute ROC curve and ROC area for each class
    # y_label为测试集的正确标签，y_score为模型预测的测试集得分
    fpr, tpr, threshold = roc_curve(y_label, y_score)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def ROC_nClass(y_label, y_score, n_classes):
    '''
    :param y_label: one-hot 编码的测试集正确标签
    :param y_score: 测试样本每一类的得分
    :param n_classes：分类结果的类型数目
    :return: None
    '''
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green',
                    'blue', 'purple', 'yellow', 'pink', 'lightgoldenrodyellow',
                    'chocolate', 'darkorchid'])
    # zip是python3中以迭代器的形式返回元组
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    train(3)
    test()
    show_sample(train_dataset, train_label)
    print("debugging")