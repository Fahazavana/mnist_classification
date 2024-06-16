import torch
from tqdm import tqdm
import numpy as np

##############################################################################
# MLP
##############################################################################


def eval_mlp(model, criterion, data, device='cpu', per_class=False):
    n = len(data.dataset)
    pred_perclass = [0 for x in range(10)]
    total_perclass = [0 for x in range(10)]
    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        for images, labels in data:
            y_hat = model(images.reshape(-1, 28*28).to(device))
            labels = labels.to(device)
            loss += criterion(y_hat, labels).item()
            _, preds = torch.max(y_hat, 1)
            correct += (preds == labels).sum().item()

            if per_class:
                for label, pred in zip(labels, preds):
                    if label == pred:
                        pred_perclass[label.item()] += 1
                    total_perclass[label.item()] += 1

    if per_class:
        for idx, label in enumerate(pred_perclass):
            pred_perclass[idx] = 100 * pred_perclass[idx]/total_perclass[idx]
        return loss, correct/n,  pred_perclass

    return loss, 100*correct/n


def train_mlp(model, criterion, optimizer, epochs, train, device, writer,  valid):
    N = len(train.dataset)
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train), total=len(train),
                    desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        correct = 0
        running_loss = 0.0
        for k, (x, y) in pbar:
            x = x.reshape(-1, 28*28).to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                # training
                _, pred = torch.max(y_hat, 1)
                correct += (pred == y).sum().item()
            pbar.set_postfix(loss=f'{running_loss:.3f}',
                             acc=f'{100*(correct/N):.3f}')

        val_loss, val_acc = eval_mlp(model, criterion, valid, device)
        writer.add_scalar("Accuracy", 100*(correct/N),
                          global_step=epoch, new_style=True)
        writer.add_scalar("Accuracy Validation ", val_acc,
                          global_step=epoch, new_style=True)
        writer.add_scalar("Loss", running_loss,
                          global_step=epoch, new_style=True)
        writer.add_scalar("Loss Validation ", val_loss,
                          global_step=epoch, new_style=True)

        print(
            f"Validation loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}")

##############################################################################
# CNN NO MLP
##############################################################################


def eval_cnn(model, criterion, data, device='cpu', per_class=False):
    n = len(data.dataset)
    pred_perclass = [0 for x in range(10)]
    total_perclass = [0 for x in range(10)]
    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        for images, labels in data:
            y_hat = model(images.to(device))
            labels = labels.to(device)
            loss += criterion(y_hat, labels).item()
            _, preds = torch.max(y_hat, 1)
            correct += (preds == labels).sum().item()
            if per_class:
                for label, pred in zip(labels, preds):
                    if label == pred:
                        pred_perclass[label.item()] += 1
                    total_perclass[label.item()] += 1

    if per_class:
        for idx, label in enumerate(pred_perclass):
            pred_perclass[idx] = 100 * pred_perclass[idx]/total_perclass[idx]
        return loss, correct/n,  pred_perclass

    return loss, 100*correct/n


def train_cnn(model, criterion, optimizer, epochs, train, device, writer,  valid):
    N = len(train.dataset)
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train), total=len(train),
                    desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        correct = 0
        running_loss = 0.0
        for k, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                # training
                _, pred = torch.max(y_hat, 1)
                correct += (pred == y).sum().item()
            pbar.set_postfix(loss=f'{running_loss:.3f}',
                             acc=f'{100*(correct/N):.3f}')

        val_loss, val_acc = eval_cnn(model, criterion, valid, device)
        writer.add_scalar("Accuracy", 100*(correct/N),
                          global_step=epoch, new_style=True)
        writer.add_scalar("Accuracy Validation ", val_acc,
                          global_step=epoch, new_style=True)
        writer.add_scalar("Loss", running_loss,
                          global_step=epoch, new_style=True)
        writer.add_scalar("Loss Validation ", val_loss,
                          global_step=epoch, new_style=True)

        print(
            f"Validation loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}")
