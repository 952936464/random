import math
import torch
from torchvision import models
import torch.nn.functional as F
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # Flatten data from (n,1,28,28) to (n,784) Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
        x = x.view(batch_size, -1)
        x = self.fc(x)  # x应用全连接网络
        return x

# 各种机器学习模型
def get_model(name="cnn", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    elif name == "cnn":
        model = Net()

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
def model_norm(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)
def model_cos(model_1, model_2):
    sum = 0
    sum1 = 0
    sum2 = 0
    cos = 0
    for name, layer in model_1.named_parameters():
        sum += torch.sum(layer.data * model_2.state_dict()[name].data)
        sum1 += torch.sum(layer.data * layer.data)
        sum2 += torch.sum(model_2.state_dict()[name].data * model_2.state_dict()[name].data)
        cos = sum / (math.sqrt(sum1) * math.sqrt(sum2))
    return max(0,cos)