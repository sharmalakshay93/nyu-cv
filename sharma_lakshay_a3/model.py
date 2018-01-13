import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*5*5, 128)
        self.fc2 = nn.Linear(128, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10*4*4, 10*4*4)
        self.fc2 = nn.Linear(10*4*4, 100)
        self.fc3 = nn.Linear(100, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 10*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

# class Net3(nn.Module):
#     def __init__(self):
#         super(Net3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
#         self.conv3 = nn.Conv2d(128, 512, kernel_size=3)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(40*5*5, 200)
#         self.fc2 = nn.Linear(200, 100)
#         self.fc3 = nn.Linear(100, nclasses)


#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
#         x = x.view(-1, 40*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return F.log_softmax(x)

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*11*11, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3_drop(self.conv3(x)))
        x = x.view(-1, 128*11*11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*12*12, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_drop(F.max_pool2d(self.conv3(x), 2)))
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*12*12, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_drop(F.max_pool2d(self.conv3(x), 2)))
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*12*12, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_drop(F.max_pool2d(self.conv3(x), 2)))
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

# class Net8(nn.Module):
#     def __init__(self):
#         super(Net8, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(128*12*12, 5000)
#         self.fc2 = nn.Linear(5000, 1000)
#         self.fc3 = nn.Linear(1000, nclasses)


#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 128*12*12)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc3(x)
#         return F.log_softmax(x)

class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*12*12, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

class Net9(nn.Module):
    def __init__(self):
        super(Net9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(192*11*11, 11616)
        self.fc2 = nn.Linear(11616, 5808)
        self.fc3 = nn.Linear(5808, 2904)
        self.fc4 = nn.Linear(2904, nclasses)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 192*11*11)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        x = self.fc4(x)
        return F.log_softmax(x)

class Net10(nn.Module):
    def __init__(self):
        super(Net10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*13*13, 256)
        self.fc2 = nn.Linear(256, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 128*13*13)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net11(nn.Module):
    def __init__(self):
        super(Net11, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(192*13*13, 512)
        self.fc2 = nn.Linear(512, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 192*13*13)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128*12*12, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

class Net13(nn.Module):
    # based on Net10
    def __init__(self):
        super(Net13, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
        self.fc1 = nn.Linear(128*13*13, 256)
        self.fc2 = nn.Linear(256, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128*13*13)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net14(nn.Module):
    # based on Net10 and GTRSB winner
    def __init__(self):
        super(Net14, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class Net15(nn.Module):
    # like Net14, but has regularization
    def __init__(self):
        super(Net15, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net16(nn.Module):
    # like Net15, but has dropout at conv2
    def __init__(self):
        super(Net16, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net17(nn.Module):
    # like Net15, but has dropout at conv3
    def __init__(self):
        super(Net17, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net18(nn.Module):
    # like Net16 and Net17, but has dropout at conv2 and conv3
    def __init__(self):
        super(Net18, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net19(nn.Module):
    # like Net17, but wider at conv1
    def __init__(self):
        super(Net19, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net20(nn.Module):
    # like Net19, but with another layer of dropout (like model18)
    def __init__(self):
        super(Net20, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net21(nn.Module):
    # like Net18, but with dropout at conv1 and conv2 (instead of conv2 and conv3)
    def __init__(self):
        super(Net21, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net22(nn.Module):
    # like Net18, but with dropout at all conv layers
    def __init__(self):
        super(Net22, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net23(nn.Module):
    # like Net18, but has an extra fc layer
    def __init__(self):
        super(Net23, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net24(nn.Module):
    # like Net22, but trained for 30 epochs
    def __init__(self):
        super(Net24, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net25(nn.Module):
    # like Net23, but 30 training epochs
    def __init__(self):
        super(Net25, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net26(nn.Module):
    # like Net24 and 25, with features of both (all conv have dropout, and 2 fcs)
    def __init__(self):
        super(Net26, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net27(nn.Module):
    # like Net25, but with a wider conv layer 30 training epochs
    def __init__(self):
        super(Net27, self).__init__()
        self.conv1 = nn.Conv2d(3, 40, kernel_size=7)
        self.conv2 = nn.Conv2d(40, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net28(nn.Module):
    # like Net25, but wider conv layer (smaller than net27) 30 training epochs
    def __init__(self):
        super(Net28, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=7)
        self.conv2 = nn.Conv2d(30, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net29(nn.Module):
    # like Net23, extra conv layer (no padding) and 30 training epochs
    def __init__(self):
        super(Net29, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*3*3, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256*3*3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net30(nn.Module):
    # like Net29, but with padding
    def __init__(self):
        super(Net30, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net31(nn.Module):
    # like Net29, but with dropout (no padding)
    def __init__(self):
        super(Net31, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*3*3, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(self.conv4_drop(self.conv4(x)))
        x = x.view(-1, 256*3*3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net32(nn.Module):
    # like Net30, but with dropout and padding
    def __init__(self):
        super(Net32, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(self.conv4_drop(self.conv4(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net33(nn.Module):
    # like Net25, but with batchnorm only on conv2
    def __init__(self):
        super(Net33, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net34(nn.Module):
    # like Net25/Net33, but with batchnorm on conv2 and conv3
    def __init__(self):
        super(Net34, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net35(nn.Module):
    # like Net25/Net33, but with batchnorm on conv1 and conv2
    def __init__(self):
        super(Net35, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)


class Net36(nn.Module):
    # like Net25/Net33, but with batchnorm on conv1 and conv3
    def __init__(self):
        super(Net36, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net37(nn.Module):
    # like Net25/Net33, but with batchnorm on conv1, conv2, conv3
    def __init__(self):
        super(Net37, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net38(nn.Module):
    # like Net25/Net33, but with batchnorm on conv1, conv2, conv3
    def __init__(self):
        super(Net38, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net39(nn.Module):
    # like Net25/Net33, but with batchnorm on conv1
    def __init__(self):
        super(Net39, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net40(nn.Module):
    # like Net36, but wider conv layers
    def __init__(self):
        super(Net40, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net41(nn.Module):
    # like Net40, but wider fc2 layers
    def __init__(self):
        super(Net41, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net42(nn.Module):
    # like Net40, but wider fc2 layers
    def __init__(self):
        super(Net42, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net43(nn.Module):
    # like Net40, but wider fc2 layers
    def __init__(self):
        super(Net43, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1500)
        self.fc2 = nn.Linear(1500, 700)
        self.fc3 = nn.Linear(700, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net44(nn.Module):
    # like Net43, but with dropout 
    def __init__(self):
        super(Net44, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_drop = nn.Dropout2d(p=0.3)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1500)
        self.fc2 = nn.Linear(1500, 700)
        self.fc3 = nn.Linear(700, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1_bn(self.conv1(x))), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net45(nn.Module):
    # like Net36, but with 50 epochs
    def __init__(self):
        super(Net45, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 256*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net46(nn.Module):
    # like Net42, but wider fc2 layers
    def __init__(self):
        super(Net46, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net47(nn.Module):
    # like Net46, but pre-trained and smaller lr
    def __init__(self):
        super(Net47, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net48(nn.Module):
    # like Net46, but pre-trained and 100 epochs and adam optim
    def __init__(self):
        super(Net48, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net49(nn.Module):
    # VGG19 not pretrained
    def __init__(self):
        super(Net49, self).__init__()
        self.vgg19 = models.vgg19_bn(pretrained=False)
        self.replaceLayer()

    def replaceLayer(self):
        self.vgg19.classifier._modules['6'] = nn.Linear(4096, 43)

    def forward(self, x):
        x = self.vgg19(x)
        return F.log_softmax(x)

# class Net50(nn.Module):
#     # VGG19 not pretrained
#     def __init__(self):
#         super(Net49, self).__init__()
#         self.vgg19 = models.vgg19_bn(pretrained=True)
#         self.replaceLayer()

#     def replaceLayer(self):
#         for param in self.vgg19.parameters():
#             param.requires_grad = False

#         self.vgg19.classifier._modules['6'] = nn.Linear(4096, 43)

    def forward(self, x):
        x = self.vgg19(x)
        return F.log_softmax(x)


class Net50(nn.Module):
    #is Net2 with 64 input size
    def __init__(self):
        super(Net50, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*13*13, 128)
        self.fc2 = nn.Linear(128, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 128*13*13)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net51(nn.Module):
    # like Net46, but pre-trained and smaller lr (untested)
    def __init__(self):
        super(Net51, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 150, kernel_size=6)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(150, 300, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(300)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(300*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, nclasses)
        self.set_grad()
        self.fc4 = nn.Linear(nclasses, nclasses)

    def set_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3_bn(self.conv3(x))), 2))
        x = x.view(-1, 300*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

class Net52(nn.Module):
    # load state dict    
    def __init__(self):
        super(Net52, self).__init__()
        self.net47 = self.get_pretrained()
        self.set_grad()
        self.fc4 = nn.Linear(43,43)
        
    def get_pretrained(self):
        state_dict = torch.load('./saved_models/Net47/model_2.pth')
        model = Net47()
        model.cuda()
        model.load_state_dict(state_dict)
        return model
        
    def set_grad(self):
        for param in self.net47.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.net47(x))
        x = self.fc4(x)
        return F.log_softmax(x)








