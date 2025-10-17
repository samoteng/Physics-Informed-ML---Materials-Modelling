import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py

# Loss Function
class Lossfunc(object):
    def __init__(self):
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss

    def __call__(self, output, target):
        return self.criterion(output, target)

# Data Reader for Material C
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path, 'r')

    def get_strain(self):
        # For Material C, original shape in file: [50, 1, 1100]
        # Transpose to [1100, 50, 1] --> [samples, steps, directions]
        strain = np.array(self.data['strain']).transpose(2, 0, 1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        # For Material C, original shape in file: [50, 1, 1100]
        # Transpose to [1100, 50, 1]
        stress = np.array(self.data['stress']).transpose(2, 0, 1)
        return torch.tensor(stress, dtype=torch.float32)

# Data Normalizer
class DataNormalizer(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        # data is expected to be 2D: [n_samples, features]
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

# Neural Network Model
class Const_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, output_dim):

        super(Const_Net, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        layers = []
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer (linear activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Main Script
if __name__ == '__main__':
    # Data Loading for Material C
    data_path = 'Problem_1_student/Data/Material_C.mat'  # update the path if needed
    data_reader = MatRead(data_path)
    strain = data_reader.get_strain()  # shape: [1100, 50, 1]
    stress = data_reader.get_stress()  # shape: [1100, 50, 1]

    # Get number of time steps and directions from data shape
    n_steps = strain.shape[1]      # should be 50
    n_directions = strain.shape[2]   # should be 1 for Material C

    # Data Splitting (70:15:15)
    num_samples = strain.shape[0]
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))
    n_train = int(0.7 * num_samples)
    n_val = int(0.15 * num_samples)
    n_test = num_samples - n_train - n_val

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train+n_val]
    test_indices  = indices[n_train+n_val:]

    train_strain = strain[train_indices]
    train_stress = stress[train_indices]
    val_strain   = strain[val_indices]
    val_stress   = stress[val_indices]
    test_strain  = strain[test_indices]
    test_stress  = stress[test_indices]

    # Flatten the tensors for the FCNN: each sample becomes a vector
    # For Material C: 50 time steps x 1 direction = 50 features
    train_strain = train_strain.view(train_strain.size(0), -1)
    train_stress = train_stress.view(train_stress.size(0), -1)
    val_strain   = val_strain.view(val_strain.size(0), -1)
    val_stress   = val_stress.view(val_stress.size(0), -1)
    test_strain  = test_strain.view(test_strain.size(0), -1)
    test_stress  = test_stress.view(test_stress.size(0), -1)

    # Data Normalization
    strain_normalizer = DataNormalizer()
    strain_normalizer.fit(train_strain)
    train_strain_enc = strain_normalizer.transform(train_strain)
    val_strain_enc   = strain_normalizer.transform(val_strain)
    test_strain_enc  = strain_normalizer.transform(test_strain)

    stress_normalizer = DataNormalizer()
    stress_normalizer.fit(train_stress)
    train_stress_enc = stress_normalizer.transform(train_stress)
    val_stress_enc   = stress_normalizer.transform(val_stress)
    test_stress_enc  = stress_normalizer.transform(test_stress)

    # Create DataLoaders
    batch_size = 20
    train_dataset = Data.TensorDataset(train_strain_enc, train_stress_enc)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation DataLoader
    val_dataset = Data.TensorDataset(val_strain_enc, val_stress_enc)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define Neural Network
    input_dim = train_strain_enc.size(1)   # 50 for Material C
    output_dim = train_stress_enc.size(1)    # 50 for Material C
    net = Const_Net(input_dim=input_dim, hidden_dim=70, num_hidden_layers=1, output_dim=output_dim)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters: %d' % n_params)

    # Define Loss Function, Optimizer, and Scheduler
    loss_func = Lossfunc()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training Loop
    epochs = 350
    loss_train_list = []
    loss_val_list = []

    print("Start training for {} epochs...".format(epochs))
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_inputs)
            loss = loss_func(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Compute validation loss
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                outputs = net(val_inputs)
                loss = loss_func(outputs, val_targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        loss_train_list.append(train_loss)
        loss_val_list.append(val_loss)

    print("Final Train Loss: {:.6f}".format(train_loss))
    print("Final Validation Loss: {:.6f}".format(val_loss))

    # Plotting Train and Validation Loss vs Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(loss_train_list, label='Train Loss')
    plt.plot(loss_val_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train and Validation Loss vs Epochs (Material C)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.2)  # Set y-axis range
    plt.tight_layout()

    # Plot a Sample: True Stress vs Predicted Stress
    net.eval()
    with torch.no_grad():
        sample_input = test_strain_enc[0:1]   # Use the first sample from the test set
        sample_true  = test_stress_enc[0:1]     # Normalized true stress
        sample_pred  = net(sample_input)        # Predicted normalized stress

    # Inverse transform to get the original scale
    sample_pred_denorm = stress_normalizer.inverse_transform(sample_pred)
    sample_true_denorm = stress_normalizer.inverse_transform(sample_true)

    # Reshape to [time_steps, n_directions] (for Material C: [50, 1])
    sample_pred_curve = sample_pred_denorm.view(-1, n_directions).cpu().numpy()
    sample_true_curve = sample_true_denorm.view(-1, n_directions).cpu().numpy()

    plt.figure(figsize=(8, 5))
    time_steps = np.linspace(0, 1, sample_true_curve.shape[0])
    plt.plot(time_steps, sample_true_curve[:, 0], 'o-', label=r"True $\sigma_{11}$")
    plt.plot(time_steps, sample_pred_curve[:, 0], 'x--', label=r"Predicted $\sigma_{11}$")
    plt.xlabel('Normalized Time')
    plt.ylabel('Stress')
    plt.title('Truth vs Approximate Stress for One Sample (Material C)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Plot a Sample: Stress vs Strain Curve
    sample_strain_denorm = strain_normalizer.inverse_transform(sample_input)
    sample_strain_curve = sample_strain_denorm.view(-1, n_directions).cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(sample_strain_curve[:, 0], sample_true_curve[:, 0], 'o-', label=r"True $\sigma_{11}$ vs $\varepsilon_{11}$")
    plt.plot(sample_strain_curve[:, 0], sample_pred_curve[:, 0], 'x--', label=r"Predicted $\sigma_{11}$ vs $\varepsilon_{11}$")
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Stress vs Strain for One Sample (Material C)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
