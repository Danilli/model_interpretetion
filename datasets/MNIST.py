from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

def get_trainset():
    trainset = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    return trainset


def get_testset():
    testset = MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    return testset

def print_image(trainset, img_num = 0):
    image, label = trainset[img_num]
    print(f"Image shape: {image.shape}")
    plt.imshow(image.squeeze())
    plt.title(label)