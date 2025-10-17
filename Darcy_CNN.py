import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py


# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        # print('x.shape',x.shape)
        # print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)


# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)


# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x


# Define network
class CNN(nn.Module):
    def __init__(self, channel_width=64):
        super(CNN, self).__init__()
        # CNN with five convolutional layers.
        # Using kernel_size=3 and padding=1
        # ReLU activations add nonlinearity to capture the complex mapping from a to u.
        self.layers = nn.Sequential(
            nn.Conv2d(1, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, channel_width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel_width, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layers(x)
        out = out.squeeze(1)
        return out


if __name__ == '__main__':
    ############################# Data processing #############################
    # Read data from .mat files
    train_path = 'Darcy_2D_data_train.mat'
    test_path = 'Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a()
    u_train = data_reader.get_u()

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a()
    u_test = data_reader.get_u()

    # Normalize data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train)
    a_test = a_normalizer.encode(a_test)

    u_normalizer = UnitGaussianNormalizer(u_train)

    print("a_train shape:", a_train.shape)
    print("a_test shape:", a_test.shape)
    print("u_train shape:", u_train.shape)
    print("u_test shape:", u_test.shape)

    # Create data loader
    batch_size = 20
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    ############################# Define and train network #############################
    # Create RNN instance, define loss function, optimizer and scheduler
    channel_width = 64
    net = CNN(channel_width=channel_width)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    loss_func = LpLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Train network
    epochs = 200  # Number of epochs
    print("Start training CNN for {} epochs...".format(epochs))
    start_time = time()

    loss_train_list = []
    loss_test_list = []
    x_axis = []
    for epoch in range(epochs):
        net.train()
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            output = net(input)  # Forward pass
            output = u_normalizer.decode(output)
            l = loss_func(output, target)  # Compute loss

            optimizer.zero_grad()  # Clear gradients
            l.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            scheduler.step()  # Update learning rate

            trainloss += l.item()

        # Test
        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test).item()

        # Print train loss every 10 epochs
        if epoch % 10 == 0:
            print("Epoch: {}, Train loss: {:.5f}, Test loss: {:.5f}".format(epoch, trainloss / len(train_loader),
                                                                            testloss))

        loss_train_list.append(trainloss / len(train_loader))
        loss_test_list.append(testloss)
        x_axis.append(epoch)

    total_time = time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time: {}'.format(total_time_str))
    print("Final Train loss: {:.5f}".format(trainloss / len(train_loader)))
    print("Final Test loss: {:.5f}".format(testloss))

    # Plot loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, loss_train_list, label='Train loss')
    plt.plot(x_axis, loss_test_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.05)
    plt.legend()
    plt.grid()
    plt.title('Train and Test Loss vs. Epochs')
    plt.show()

    # Plot contour plots for a test sample
    net.eval()
    with torch.no_grad():
        sample_idx = 0
        a_sample = a_test[sample_idx:sample_idx + 1]
        u_pred = net(a_sample)
        u_pred = u_normalizer.decode(u_pred)
        u_true = u_test[sample_idx].cpu().numpy()
        u_pred = u_pred.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(u_true, levels=50)
    plt.title("True solution")
    plt.colorbar(cp1)

    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(u_pred, levels=50)
    plt.title("Predicted solution")
    plt.colorbar(cp2)

    plt.suptitle("Contour Plots for Darcy 2D Problem")
    plt.show()
