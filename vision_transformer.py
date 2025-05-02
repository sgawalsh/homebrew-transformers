import torch, pickle, pickle, os, set_device
from model import TransformerEncoderBlock
from torch import nn
import numpy as np
from torch.nn import functional as F
from math import floor, sqrt
from torch.utils.tensorboard import SummaryWriter


class cnnClassifier(nn.Module): #simple convolutional model to compare against
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VisionTransformerEncoder(nn.Module):  #@save
    """The Transformer encoder."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X):
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, None)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class visionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=128, mlp_dim=256, depth=6, heads=4, dropout = 0.1, classTokenMode: bool = False): # (dim=128, mlp_dim=256, lr= 0.001)
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.classTokenMode = classTokenMode
        
        if classTokenMode:
            self.classToken = nn.Parameter(torch.zeros(1, 1, dim))

        # Patch embedding
        self.patch_to_embedding = nn.LazyLinear(dim)
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches, dim))

        # Transformer
        self.transformer = VisionTransformerEncoder(dim, mlp_dim, heads, depth, dropout)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        # Prepare patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # extract patches along rows and columns
        patches = patches.permute(0, 2, 3, 4, 5, 1) # channels to last dimension
        patches = patches.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size * 3).to(torch.float32) # group patches

        # Embed patches and add positional embedding
        x = self.patch_to_embedding(patches) + self.positional_embedding

        if self.classTokenMode:
            x = torch.cat((self.classToken.expand(patches.size(0), -1, -1), x), 1) # concat class token and patches

        # Pass through transformer
        x = self.transformer(x)

        # Classification
        x = x[:, 0, :].squeeze() if self.classTokenMode else x.mean(dim=1)
        return self.mlp_head(x) # to class dim

    def _getPatches(self, x):
        patchesBatch = torch.zeros((x.size()[0], 64, 48))
        for batchNum in range(x.size()[0]):
            for i in range(0, 64):
                col = (i * 4) % 32
                row = floor((i * 4) / 32) * 4
                # if batchNum == 0:
                #     print(row, col)
                newPatch = x[batchNum, row:row + 4, col:col+4, :]
                patchesBatch[batchNum, i] = newPatch.reshape((48))
        return patchesBatch

def stitchPatches(patches): # debugging fold
    dim = int(sqrt(len(patches)))
    patchDim = int(sqrt(len(patches[0]) / 3))
    img = np.zeros((dim * patchDim, dim * patchDim, 3))

    for i, patch in enumerate(patches):
        row = floor(i / dim) * 4
        col = (i % dim) * 4
        img[row:row + 4, col:(col + 4)] = patch.reshape(patchDim, patchDim, 3)

    # plt.imshow(img.reshape(32, 32, 3))
    # plt.show()

def getBatch(i, folderName= "//cifar-10-batches-py//", fileRootName = "data_batch_"):
    fileName = os.getcwd() + folderName + fileRootName + str(i)

    with open(fileName, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict[b'labels'], torch.from_numpy(dict[b'data'].reshape(-1, 3, 32, 32) / 255) # normalize and reshape to (N, 3, 32, 32)

def doubleBatch(labels, data, n=1):
    l = len(labels)
    assert(l == len(data))
    for ndx in range(0, l, n):
        yield data[ndx:min(ndx + n, l)], labels[ndx:min(ndx + n, l)]

def batchLoop(bLabels, bData, miniBatchLength, model, device, criterion, optim, e, epochs, runningLoss, runningAccuracy, count, isTrain):
    for x, y in doubleBatch(torch.tensor(bLabels).to(device), bData, miniBatchLength):
        yHat = model(x.to(device).to(torch.float32))
        loss = criterion(yHat, y)
        runningLoss += loss.item()
        runningAccuracy += (yHat.argmax(1) == y).float().mean()

        if isTrain:
            optim.zero_grad()
            loss.backward()
            optim.step()
        count +=1
        print(f'{e}/{epochs} - Running Loss: {runningLoss / count:.3f} - Loss: {loss.item():.3f} - Accuracy: {runningAccuracy * 100 / count:.1f}')
    
    return runningLoss, runningAccuracy, count


def train(miniBatchLength = 128, lr = 0.001, epochs = 10, classTokenMode = False):
    device = set_device.device
    model = visionTransformer(classTokenMode=classTokenMode).to(device)
    # model = cnnClassifier().to(device) # compare against cnn
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(f"runs/{"classToken" if classTokenMode else "mean"}")

    for e in range(1, epochs + 1):
        
        runningLoss, runningAccuracy = 0.0, 0.0
        count = 0

        for i in range(1, 5): # training
            bLabels, bData = getBatch(i)
            
            runningLoss, runningAccuracy, count = batchLoop(bLabels, bData, miniBatchLength, model, device, criterion, optim, e, epochs, runningLoss, runningAccuracy, count, isTrain=True)

        writer.add_scalar("Loss/train", runningLoss / count, e)
        writer.add_scalar("Accuracy/train", runningAccuracy / count, e)

    bLabels, bData = getBatch(5) # evaluation
    runningLoss, runningAccuracy = 0.0, 0.0
    count = 0
    runningLoss, runningAccuracy, count = batchLoop(bLabels, bData, miniBatchLength, model, device, criterion, optim, e, epochs, runningLoss, runningAccuracy, count, isTrain=False)
    writer.add_scalar("Loss/eval", runningLoss / count, 0)
    writer.add_scalar("Accuracy/eval", runningAccuracy / count, 0)
    writer.close()

train(epochs = 10, classTokenMode=True)
