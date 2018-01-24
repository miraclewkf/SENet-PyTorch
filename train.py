from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from SENet import *
import argparse
from read_ImageNetData import ImageNetData

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i+1)*args.batch_size)
                batch_acc = running_corrects / ((i+1)*args.batch_size)

                if phase == 'train' and i%args.print_freq == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}]  {} Loss: {:.4f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, phase, batch_loss, batch_acc, \
                          args.print_freq / (time.time() - tic_batch)))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        if (epoch+1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="For training from one checkpoint at start epoch")
    parser.add_argument('--pretrain', type=str, default="", help="For training from one pretrain model")
    args = parser.parse_args()

    # read data
    dataloders, dataset_sizes = ImageNetData(args)

    # use gpu or not
    use_gpu = torch.cuda.is_available()

    # get model
    model = senet50()

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(("=> loading pretrain model '{}'".format(args.pretrain)))
            checkpoint = torch.load(args.pretrain)
            base_dict = {".".join(["module", k]): v for k, v in list(checkpoint.items())}
            model.load_state_dict(base_dict, strict=False)
        else:
            print(("=> no pretrain model found at '{}'".format(args.pretrain)))

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    # replace the original fc layer with your fc layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_class)

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs,
                           dataset_sizes=dataset_sizes)