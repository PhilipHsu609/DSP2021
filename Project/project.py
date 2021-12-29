import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import Dataset, DataLoader

# Setting seed
seed = 7777
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.manual_seed(seed)

device = 'cpu'
if torch.cuda.is_available():
    print("GPU available!")
    print("Using GPU:", torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties(0), "\n")
    torch.cuda.manual_seed_all(seed)
    device = 'cuda'

class BearingDataset(Dataset):
    def __init__(self, data, label, mode, train_ratio=0.8) -> None:
        self.mode = mode
        self.label = label

        samples = int(len(data) * train_ratio)
        if mode == "train":
            self.data = data[:samples, :, :]
            self.label = label[:samples]
        elif mode == "eval":
            self.data = data[samples:, :, :]
            self.label = label[samples:]
        else:
            self.data = data

        self.data = torch.from_numpy(np.array(self.data).astype(np.float32))
        if label is not None:
            self.label = torch.from_numpy(np.array(self.label).astype(np.int64))

    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.data[index], self.label[index]
        else:
            return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)

class Network(nn.Module):
    def __init__(self, task1=True) -> None:
        super(Network, self).__init__()

        if task1:
            self.conv = nn.Sequential(
                nn.Conv1d(2, 4, 64, 32),
                nn.ReLU(),
                nn.Conv1d(4, 8, 64, 32),
                nn.ReLU()    
            )
            self.linear = nn.Sequential(
                nn.Linear(112, 3)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(2, 1, 64, 32),
                nn.ReLU()
            )
            self.linear = nn.Sequential(
                nn.Linear(499, 3)
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def loss(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, target)
        return loss

# Loading data
traindata = np.load("./data/traindata.npy")
trainlabel = np.load("./data/trainlabel.npy")
anomalydata = np.load("./data/anomaly_sample.npy")
testdata = np.load("./data/anomalytestdata.npy")

# Standardization
ss = StandardScaler()
ss.fit(traindata[:, 0, :])
traindata[:, 0, :] = ss.transform(traindata[:, 0, :])
anomalydata[:, 0, :] = ss.transform(anomalydata[:, 0, :])
testdata[:, 0, :] = ss.transform(testdata[:, 0, :])

ss.fit(traindata[:, 1, :])
traindata[:, 1, :] = ss.transform(traindata[:, 1, :])
anomalydata[:, 1, :] = ss.transform(anomalydata[:, 1, :])
testdata[:, 1, :] = ss.transform(testdata[:, 1, :])

# Shuffle training data
p = np.random.permutation(len(traindata))
traindata = traindata[p]
trainlabel = trainlabel[p]

# Hyperparameters for neural network
n_epochs = 50
batch_size = 32
lr = 5e-3
task2_model_path = "./weight_one_conv.ckpt"

# Prepare datasets
train_set = BearingDataset(traindata, trainlabel, "train")
validate_set = BearingDataset(traindata, trainlabel, "eval")
train_ldr = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_ldr = DataLoader(validate_set, batch_size=batch_size, shuffle=False)

# Using single cnn layer model
model = Network(task1=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_record = {"train": [], "eval": []}
acc_record = {"train": [], "eval": []}

# Training
for epoch in range(n_epochs):
    model.train()
    train_loss, train_acc = 0, 0
    for x, y in train_ldr:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = model.loss(pred, y)
        loss.backward()
        optimizer.step()

        train_acc += (pred.argmax(1).cpu() == y.cpu()).sum().item()
        train_loss += loss.item()

    train_acc /= len(train_ldr.dataset)
    train_loss /= len(train_ldr)
    acc_record["train"].append(train_acc)
    loss_record["train"].append(train_loss)

    eval_loss, eval_acc = 0, 0
    model.eval()
    for x, y in val_ldr:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = model.loss(pred, y)
        eval_acc += (pred.argmax(1).cpu() == y.cpu()).sum().item()
        eval_loss += loss.item()
    eval_acc /= len(val_ldr.dataset)
    eval_loss /= len(val_ldr)

    acc_record["eval"].append(eval_acc)
    loss_record["eval"].append(eval_loss)

    print("train loss = {:.4f}, acc = {:.4f}, eval loss = {:.4f}, acc = {:.4f}".format(
        train_loss, train_acc, eval_loss, eval_acc
    ))
    epoch += 1

del train_set, train_ldr, validate_set, val_ldr

# Get the kernel parameters from 50th epoch
kernel = model.conv[0].weight.cpu()

def apply_kernel(data):
    x = torch.from_numpy(data)
    x = F.conv1d(x, kernel, stride=32)
    x = x.view(x.size(0), -1).detach().numpy()
    return x

# Prepare data for one class svm
train_svm = apply_kernel(traindata)
anomaly_svm = apply_kernel(anomalydata)
test_svm = apply_kernel(testdata)

param = {
    "gamma": 11 / 10000,
    "tol": 1e-3,
    "nu": 0.1,
    "cache_size": 1024
}

clf = OneClassSVM(**param)
clf.fit(train_svm)

anomaly_pred = clf.predict(anomaly_svm)
test_anomaly_pred = clf.predict(test_svm)

test_set = BearingDataset(testdata, None, "test")
test_ldr = DataLoader(test_set, batch_size=1, shuffle=False)

# Load task 1 model
task1_model_path = "./weight.ckpt"
model = Network(task1=True).to(device)
model.load_state_dict(torch.load(task1_model_path))

print("Writing result to result_anomaly.csv...")

model.eval()
with open("result_anomaly.csv", "w") as f:
    f.write("id,category\n")

    for i, x in enumerate(test_ldr):
        x = x.to(device)

        if test_anomaly_pred[i] == -1:
            # Current sample is anomaly
            pred = 3
        else:
            with torch.no_grad():
                output = model(x)
            pred = output.argmax(dim=1, keepdim=True).item()

        f.write("%d,%d\n" % (i, pred))

print("Done!")