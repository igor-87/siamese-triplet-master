from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit,get_vectors
import numpy as np

cuda = torch.cuda.is_available()
import matplotlib
import matplotlib.pyplot as plt
from datasets import BalancedBatchSampler, CustomDataset
# Set up the network and training parameters
from networks import EmbeddingNet, InceptionBased
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric
import inception as incept

# mean, std = 0.28604059698879553, 0.35302424451492237
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10

fashion_mnist_classes = [1,2,3,4,5,6,7,8,9,10]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']


def plot_embeddings(embeddings, targets):
    # plt.figure(figsize=(10, 10))
    for i in range(len(fashion_mnist_classes)):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    plt.legend(fashion_mnist_classes)


def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = np.zeros((len(dataloader.dataset), 2))
    labels = np.zeros(len(dataloader.dataset))
    k = 0
    for images, target in dataloader:
        images = Variable(images, volatile=True)
        if cuda:
            images = images.cuda()
        embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
        labels[k:k + len(images)] = target.numpy()
        k += len(images)
    return embeddings, labels
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

train_dataset = CustomDataset(csv_file='data.csv',
                              root_dir='C:\\Users\\alex\\Desktop\\pytourch\\siamese-triplet-master\\images\\',
                              transform=transforms.Compose([
                                  transforms.Scale(299),
                                  transforms.ToTensor()
                                  # normalize
                                  # transforms.Normalize((mean,), (std,))
                              ]))

test_dataset = CustomDataset(csv_file='data.csv',
                             root_dir='C:\\Users\\alex\\Desktop\\pytourch\\siamese-triplet-master\\images\\',
                             transform=transforms.Compose([
                                 transforms.Scale(299),
                                 transforms.ToTensor()
                                 # normalize
                                 # transforms.Normalize((mean,), (std,))
                             ]))
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=2)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=2)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

im_size = 299
margin = 1.
model = incept.inceptionv4(num_classes=1000, pretrained='imagenet')
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    metrics=[AverageNonzeroTripletsMetric()])

train_embeddings, train_labels =get_vectors(online_train_loader, model,  optimizer, cuda)

plot_embeddings(train_embeddings, train_labels)
plt.title('Train set')
plt.savefig('tessstttyyy.png', dpi=100)
torch.save(model, 'mytraining.pt')
# val_embeddings, val_labels = extract_embeddings(online_test_loader, model)
# plot_embeddings(val_embeddings, val_labels)
# plt.title('Test set')
