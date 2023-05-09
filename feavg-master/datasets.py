from torchvision import datasets, transforms


# 获取数据集
def get_dataset(dir, name):
    if name == 'mnist':
        # root: 数据路径
        # train参数表示是否是训练集或者测试集
        # download=true表示从互联网上下载数据集并把数据集放在root路径中
        # transform：图像类型的转
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 均值0.1307，标准差0.3081 mnist数据集默认得
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 均值0.1307，标准差0.3081 mnist数据集默认得
        ])
        train_dataset = datasets.MNIST(root=dir,
                                       train=True,
                                       download=True,
                                       transform=transform_train)
        eval_dataset = datasets.MNIST(root=dir,
                                      train=False,
                                      transform=transform_test,
                                      download=True)

    elif name == 'cifar':
        # 设置两个转换格式
        # transforms.Compose 是将多个transform组合起来使用（由transform构成的列表）
        transform_train = transforms.Compose([
            # transforms.RandomCrop： 切割中心点的位置随机选取
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset