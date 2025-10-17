import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())
    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda  = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()
    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False
    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()
    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape)-1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x
    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda
    def set_torch(self, to_torch):
        self.to_torch = to_torch
    def set_float(self, to_float):
        self.to_float = to_float


# Recurrent Neural Operator (RNO)

class RNO(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_input, layer_hidden):

        super(RNO, self).__init__()
        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j+1]))
            if j != len(layer_input) - 2:
                self.layers.append(nn.SELU())
        self.hidden_layers = nn.ModuleList()
        self.hidden_size = hidden_size
        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j+1]))
            if j != len(layer_hidden) - 2:
                self.hidden_layers.append(nn.SELU())
    def forward(self, input, output, hidden, dt):
        h0 = hidden
        h = torch.cat((output, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)
        h = h * dt + h0
        combined = torch.cat((output, (output - input)/dt, hidden), 1)
        x = combined
        for _, l in enumerate(self.layers):
            x = l(x)
        output = x.squeeze(1)
        hidden = h
        return output, hidden
    def initHidden(self, b_size):
        return torch.zeros(b_size, self.hidden_size)


# Setup and Data Loading
# Define your data path (update if needed)
TRAIN_PATH = r'viscodata_3mat.mat'

# Define train and test data parameters.
Ntotal     = 400
train_size = 300
test_start = 300
N_test     = Ntotal - test_start

# Field names in the .mat file
F_FIELD   = 'epsi_tol'
SIG_FIELD = 'sigma_tol'

# Define loss function (mean squared error)
loss_func = nn.MSELoss()

######### Preprocessing data ####################
temp = torch.zeros(Ntotal, 1)
data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(F_FIELD).contiguous().view(Ntotal, -1)
data_output = data_loader.read_field(SIG_FIELD).contiguous().view(Ntotal, -1)

# We down sample the data to a coarser grid in time. This is to help saving the training time
s = 4
data_input  = data_input[:, 0::s]
data_output = data_output[:, 0::s]

inputsize = data_input.size()[1]

# Normalize data using a global min-max normalization.
data_input_min  = torch.min(data_input)
data_input_max  = torch.max(data_input)
data_input      = (data_input - data_input_min) / (data_input_max - data_input_min)

data_output_min = torch.min(data_output)
data_output_max = torch.max(data_output)
data_output     = (data_output - data_output_min) / (data_output_max - data_output_min)

# Define train and test datasets.
x_train = data_input[0:train_size, :]
y_train = data_output[0:train_size, :]

# Define the time increment dt (assumes uniform time grid)
dt = 1.0 / (y_train.shape[1] - 1)

x_test = data_input[test_start:Ntotal, :]
y_test = data_output[test_start:Ntotal, :]
testsize = x_test.shape[0]


# Define  RNO Architecture
# Define number of hidden variables.
n_hidden = 3
# Define dimensions for the constitutive model.
input_dim  = 1
output_dim = 1

# Feedforward network the RNO
layer_input  = [input_dim + output_dim + n_hidden, 100, 100, 100, output_dim]
# Network for updating the hidden state.
layer_hidden = [output_dim + n_hidden, 50, n_hidden]

net = RNO(input_dim, n_hidden, output_dim, layer_input, layer_hidden)

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    net.cuda()


# Training Configuration

epochs    = 100
b_size    = 16
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=b_size,
    shuffle=True
)


# Training the Neural Net

T = inputsize  # number of time steps
train_err = np.zeros((epochs,))
test_err  = np.zeros((epochs,))
y_test_approx = torch.zeros(testsize, T)

for ep in range(epochs):
    scheduler.step()
    train_loss = 0.0
    for x, y in train_loader:
        current_batch = x.size(0)
        hidden = net.initHidden(current_batch)
        optimizer.zero_grad()
        y_approx = torch.zeros(current_batch, T)
        y_true = y
        y_approx[:, 0] = y_true[:, 0]
        for i in range(1, T):
            y_approx[:, i], hidden = net(x[:, i].unsqueeze(1),
                                         x[:, i-1].unsqueeze(1),
                                         hidden, dt)
        loss = loss_func(y_approx, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    with torch.no_grad():
        hidden_test = net.initHidden(testsize)
        y_test_approx[:, 0] = y_test[:, 0]
        for j in range(1, T):
            y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1),
                                                   x_test[:, j-1].unsqueeze(1),
                                                   hidden_test, dt)
        t_loss = loss_func(y_test_approx, y_test)
        test_loss = t_loss.item()
    train_err[ep] = train_loss / len(train_loader)
    test_err[ep]  = test_loss
    print(f"Epoch {ep}: Train Loss = {train_err[ep]:.6f}, Test Loss = {test_err[ep]:.6f}")

# plot training/ loss history
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), train_err, label='Train Loss')
plt.plot(range(epochs), test_err, label='Test Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Test Loss History')
plt.show()


# Plotting the Stress History for a Test Sample
sample_idx = 5
true_stress = y_test[sample_idx].cpu().numpy()
predicted_stress = y_test_approx[sample_idx].cpu().numpy()
plt.figure(figsize=(8, 5))
plt.plot(true_stress, label='True Stress History', marker='o')
plt.plot(predicted_stress, label='Predicted Stress History', marker='x')
plt.xlabel('Time Step')
plt.ylabel('Normalized Stress')
plt.title('True vs Predicted Stress History (Test Sample)')
plt.legend()
plt.grid(True)
plt.show()
