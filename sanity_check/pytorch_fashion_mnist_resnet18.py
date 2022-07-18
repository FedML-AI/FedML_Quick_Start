import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from torchvision import models, transforms
from torchvision.datasets import FashionMNIST

# Data augmentation and normalization for training
# Just normalization for validation
samples = 64  # num of sample per batch
data_transforms = {
    "train": transforms.Compose(
        [
            # transforms.Resize(28),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    ),
    "test": transforms.Compose(
        [
            # transforms.Resize(28),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    ),
}

trainset = FashionMNIST(
    root="./data", train=True, download=True, transform=data_transforms["train"]
)

trainset, valset = random_split(trainset, (50000, 10000))

trainloader = DataLoader(trainset, batch_size=samples, shuffle=True, num_workers=4)

testset = FashionMNIST(
    root="./data", train=False, download=True, transform=data_transforms["test"]
)

testloader = DataLoader(testset, batch_size=samples, shuffle=False, num_workers=4)

valloader = DataLoader(valset, batch_size=samples, shuffle=False, num_workers=4)

classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)


image_datasets = {"train": trainset, "val": valset, "test": testset}
dataloaders = {"train": trainloader, "val": valloader, "test": testloader}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    val_acc,
    val_loss,
    train_acc,
    train_loss,
    epoch,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    list = {
        "train": {"acc": train_acc, "loss": train_loss},
        "val": {"acc": val_acc, "loss": val_loss},
    }
    next = epoch
    for epoch in range(next, next + num_epochs):
        print("Epoch {}/{}".format(epoch, next + num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iteration_idx = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print(
                    "iteration_idx = {}, running_loss = {}".format(
                        iteration_idx, running_loss
                    )
                )
                iteration_idx += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            list[phase]["loss"].append(epoch_loss)
            list[phase]["acc"].append(epoch_acc.item())

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, epoch + 1


model = models.resnet18(pretrained=True)
# for param in model.parameters():
#    param.requires_grad = False


# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Lists for plotting loss and accuracy and variable to
# keep track of the epoch.
# Rerun this cell if you want to restart training to empty the lists.
epoch = 0
val_acc = []
val_loss = []
train_acc = []
train_loss = []

model, epoch = train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    val_acc,
    val_loss,
    train_acc,
    train_loss,
    epoch,
    num_epochs=15,
)


dataiter = iter(testloader)
images, labels = dataiter.next()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the network on the test images : %.4f" % (correct / total))

class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print("Accuracy of %5s : %.4f" % (classes[i], class_correct[i] / class_total[i]))


checkpoint = torch.load("./FMNIST_ResNet18_noresize.tar")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
train_loss = checkpoint["train_loss"]
train_acc = checkpoint["train_acc"]
val_loss = checkpoint["val_loss"]
val_acc = checkpoint["val_acc"]
epoch = checkpoint["epoch"]
