import torch
import torch.nn.functional as F 
import torch.optim as optim
import argparse
import time
import numpy as np
from utils import load_data, accuracy
from models import GAT
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nb_heads', type=int, default=8, 
                    help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, 
                    help='Alpha for the leaky_relu.')

args = parser.parse_args()
np.random.seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, weight_decay=args.weight_decay)

train_losses = []
train_accuracy = []
val_losses = []
val_accuracy = []

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
features, adj, labels = features.to(device), adj.to(device), labels.to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    train_losses.append(loss_train.item())
    train_accuracy.append(acc_train.cpu())
    val_losses.append(loss_val.item())
    val_accuracy.append(acc_val.cpu())

    if epoch % 50 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")

# train_accuracy = train_accuracy.cpu()
# val_accuracy = val_accuracy.cpu()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

test()