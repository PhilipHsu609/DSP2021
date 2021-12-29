import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Location of training data")
parser.add_argument("--label", help="Location of training label")
args = parser.parse_args()

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
    def __init__(self) -> None:
        super(Network, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(2, 4, 64, 32),
            nn.ReLU(),
            nn.Conv1d(4, 8, 64, 32),
            nn.ReLU()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(112, 3)
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

traindata = np.load(args.data)
trainlabel = np.load(args.label)

ss = StandardScaler()

# Standardization for channel 1
ss.fit(traindata[:, 0, :])
traindata[:, 0, :] = ss.transform(traindata[:, 0, :])

# Save the pickle for testing data
with open("./channel1_scaler.pkl", "wb") as f:
    pickle.dump(ss, f)

# Standardization for channel 2
ss.fit(traindata[:, 1, :])
traindata[:, 1, :] = ss.transform(traindata[:, 1, :])

# Save the pickle for testing data
with open("./channel2_scaler.pkl", "wb") as f:
    pickle.dump(ss, f)

# Shuffle training data
p = np.random.permutation(len(traindata))
traindata = traindata[p]
trainlabel = trainlabel[p]

# Hyperparameters
n_epochs = 50
batch_size = 32
lr = 5e-3
model_path = "./weight.ckpt"

train_set = BearingDataset(traindata, trainlabel, "train")
validate_set = BearingDataset(traindata, trainlabel, "eval")

train_ldr = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_ldr = DataLoader(validate_set, batch_size=batch_size, shuffle=False)

model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_record = {"train": [], "eval": []}
acc_record = {"train": [], "eval": []}

# Training
min_loss = 100
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

    # Validation
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

    if eval_loss < min_loss:
        min_loss = eval_loss
        print("Saving model (epoch={:2d}, loss={:.4f})".format(epoch + 1, min_loss))
        torch.save(model.state_dict(), model_path)
    acc_record["eval"].append(eval_acc)
    loss_record["eval"].append(eval_loss)

    print("train loss = {:.4f}, acc = {:.4f}, eval loss = {:.4f}, acc = {:.4f}".format(
        train_loss, train_acc, eval_loss, eval_acc
    ))
    epoch += 1