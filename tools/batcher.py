def batch_trainset(trainset, batch_size = 32):
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )
    return trainloader


def batch_trainset(testset, batch_size = 32):
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )
    return testloader

