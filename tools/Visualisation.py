import matplotlib.pyplot as plt
import numpy as np

def plot(data: np, xlabel, ylabel, st=0, ed=-1):
    fig, ax = plt.subplots(1,1,figsize=(15,7))
    # Plotting the values
    ax.plot(data[st:ed])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_batch(train_features_batch, train_labels_batch, class_names, num_in_batch = 0):
    img, label = train_features_batch[num_in_batch], train_labels_batch[num_in_batch]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis("Off")

    print(f"Image size: {img.shape}")
    print(f"Label: {label}, label size: {label.shape}")
    print(f"Index: {0}")

def mnist_chi(train_features_batch,class_names, train_labels_batch, kmeans_minibactch, nrows = 5, ncols = 5):
    plt.figure(figsize=(15, 15))
    for idx in range(25):
        plt.subplot(nrows, ncols, idx+1)

        img, label = train_features_batch[idx], train_labels_batch[idx]

        kmeans_label = kmeans_minibactch.labels_[idx]
        true_class_label = class_names[label]

        plt.imshow(img.squeeze(), cmap="gray")

        if int(true_class_label[0]) == kmeans_label:
            plt.title(f"True class: {true_class_label}, Kmeans class: {kmeans_label}", c="g")
        else:
            plt.title(f"True class: {true_class_label}, Kmeans class: {kmeans_label}", c="r")

        a=plt.gca()
        a.title.set_size(7)
        plt.axis("Off")