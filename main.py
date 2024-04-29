import torch

import datasets.MNIST as mnist
from models.AE import DAC
from models.Clasterisators import KMEANS
from tools import batcher, modelTeacher

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print("----------------- All begins --------------")

    mnist_trainset = mnist.get_trainset()
    mnist_testset = mnist.get_testset()
    class_names = mnist_trainset.classes
    mnist.print_image(mnist_trainset)

    AE_model = DAC.create_AE_model()

    trainloader = batcher.batch_trainset(mnist_trainset)
    testloader = batcher.batch_testset(mnist_testset)

    AE_model = modelTeacher.train_AE(trainloader, AE_model, device=device)

    train_features_batch, train_labels_batch = next(iter(trainloader))

    encod_data = AE_model[0](train_features_batch.to(device))

    kmeans_model = KMEANS.kmeans_init(len(class_names), batch_size=32)

    kmeans_model = KMEANS.kmeans_teach(encod_data.cpu().detach().numpy(), kmeans_model)









