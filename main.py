# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import io
import pprofile
import cProfile, pstats, sys
import pyRAPL
import psutil
import pandas as pd
import zipfile
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

pyRAPL.setup()

csv_output = pyRAPL.outputs.CSVOutput('energy_pyRAPL.csv')

@pyRAPL.measure(output=csv_output)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == '__main__':

    plt.ion()   # interactive mode

    zip = zipfile.ZipFile('hymenoptera_data.zip')
    zip.extractall()

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])



    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # gives a single float value
    cpu_per_b = psutil.cpu_percent()
    # gives an object with many fields
    # vir_mem_b = psutil.virtual_memory()
    # you can convert that object to a dictionary
    vir_mem_b = dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    vir_mem_per_b = psutil.virtual_memory().percent
    # you can calculate percentage of available memory
    mem_av_per_b = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

    profiler = pprofile.Profile()
    pr = cProfile.Profile()
    pr.enable()

    with profiler:
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=3)
        csv_output.save()
    pr.disable()
    # gives a single float value
    # profiler.print_stats()
    # Or to a file:
    profiler.dump_stats("exec_time.txt")
    cpu_per_a = psutil.cpu_percent()
    # gives an object with many fields
    # vir_mem_a = psutil.virtual_memory()
    # you can convert that object to a dictionary
    vir_mem_a = dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    vir_mem_per_a = psutil.virtual_memory().percent
    # you can calculate percentage of available memory
    mem_av_per_a = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

    # ps.print_stats()

    # CPU_table = [["CPU Usage Percent", cpu_per_b, cpu_per_a], ["Memory Used", vir_mem_b, vir_mem_a],
    #          ["Memory Used(%)", vir_mem_per_b, vir_mem_per_a], ["Memory Available(%)", mem_av_per_b, mem_av_per_a]]
    # print(tabulate(CPU_table, headers=["Metric", "Before Training", "After Training"], tablefmt="pretty"))
    with open('CPU_table.txt', 'w') as f:
        # f.write(tabulate(CPU_table, headers=["Metric", "Before Training", "After Training"], tablefmt="pretty"))
        f.write("BEFORE TRAINING:--------\n")
        f.write("CPU USAGE(%):")
        f.write(str(cpu_per_b))
        f.write("\n")
        f.write("MEMORY USE:")
        f.write(str(vir_mem_b))
        f.write("\n")
        f.write("MEMORY USE(%):")
        f.write(str(vir_mem_per_b))
        f.write("\n")
        f.write("MEMORY AVAIL(%):")
        f.write(str(mem_av_per_b))
        f.write("\n")
        f.write("\n\n\n\n")
        f.write("AFTER TRAINING:---------\n")
        f.write("CPU USAGE(%):")
        f.write(str(cpu_per_a))
        f.write("\n")
        f.write("MEMORY USE:")
        f.write(str(vir_mem_a))
        f.write("\n")
        f.write("MEMORY USE(%):")
        f.write(str(vir_mem_per_a))
        f.write("\n")
        f.write("MEMORY AVAIL(%):")
        f.write(str(mem_av_per_a))
        f.write("\n")
    f.close()

    result = io.StringIO()
    pstats.Stats(pr, stream=result).print_stats()
    result = result.getvalue()
    # chop the string into a csv-like buffer
    result = 'ncalls' + result.split('ncalls')[-1]
    result = '\n'.join([','.join(line.rstrip().split(None, 5)) for line in result.split('\n')])
    # save it to disk

    with open('memory_logs.csv', 'w+') as f:
        # f=open(result.rsplit('.')[0]+'.csv','w')
        f.write(result)
        f.close()

    # # ps = pstats.Stats(pr, stream=sys.stdout)
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # # ps.print_stats()
    #
    # with open('memory_logs.txt', 'w+') as f:
    #     f.write(s.getvalue())
    # f.close()

    # data_cpu = pd.read_csv("energy_pyRAPL.csv")
    # Preview the first 5 lines of the loaded data
    # data_cpu.head()

    visualize_model(model_ft)
