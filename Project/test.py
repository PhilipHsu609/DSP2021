import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Location of testing data")
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

testdata = np.load(args.data)

with open("./channel1_scaler.pkl", "rb") as f:
    ss = pickle.load(f)
    testdata[:, 0, :] = ss.transform(testdata[:, 0, :])

with open("./channel2_scaler.pkl", "rb") as f:
    ss = pickle.load(f)
    testdata[:, 1, :] = ss.transform(testdata[:, 1, :])

test_set = BearingDataset(testdata, None, "test")
test_ldr = DataLoader(test_set, batch_size=1, shuffle=False)

model_path = "./weight.ckpt"
model = Network().to(device)
model.load_state_dict(torch.load(model_path))

print("Writing result to result.csv...")

model.eval()
with open("result.csv", "w") as f:
    f.write("id,category\n")

    for i, x in enumerate(test_ldr):
        x = x.to(device)

        with torch.no_grad():
            output = model(x)
        pred = output.argmax(dim=1, keepdim=True)

        f.write("%d,%d\n" % (i, pred.item()))

print("Done!")

