import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.tri as mtri


# Define Neural Network
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


############################# Data Processing #############################
# Load data from .mat file
path = 'Plate_data.mat'
data = scipy.io.loadmat(path)
torch.set_default_tensor_type(torch.DoubleTensor)
L_boundary = torch.tensor(data['L_boundary'], dtype=torch.float64)
R_boundary = torch.tensor(data['R_boundary'], dtype=torch.float64)
T_boundary = torch.tensor(data['T_boundary'], dtype=torch.float64)
B_boundary = torch.tensor(data['B_boundary'], dtype=torch.float64)
C_boundary = torch.tensor(data['C_boundary'], dtype=torch.float64)
Boundary = torch.tensor(data['Boundary'], dtype=torch.float64, requires_grad=True)

# truth solution from FEM
disp_truth = torch.tensor(data['disp_data'], dtype=torch.float64)

# Connectivity matrix for plotting (not used in PINN)
t_connect = torch.tensor(data['t'].astype(float), dtype=torch.float64)

# All Collocation points
x_full = torch.tensor(data['p_full'], dtype=torch.float64, requires_grad=True)

# collocation points excluding the boundary
x = torch.tensor(data['p'], dtype=torch.float64, requires_grad=True)

# This chooses 50 fixed points from the truth solution, which we will use for part (e)
rand_index = torch.randint(0, len(x_full), (50,))
disp_fix = disp_truth[rand_index, :]

# Define Neural Networks for displacement and stress
Disp_layer = [2, 300, 300, 2]
Stress_layer = [2, 400, 400, 3]

disp_net = DenseNet(Disp_layer, nn.Tanh)
stress_net = DenseNet(Stress_layer, nn.Tanh)

# Define material properties
E = 10.0
mu = 0.3

# Hooke's law for plane stress (constitutive tensor in matrix form)
stiff = E / (1 - mu ** 2) * torch.tensor([[1, mu, 0],
                                          [mu, 1, 0],
                                          [0, 0, (1 - mu) / 2]], dtype=torch.float64)
stiff = stiff.unsqueeze(0)  # shape: [1, 3, 3]

# Training parameters
iterations = 75000
loss_func = nn.MSELoss()
stiff_bc = stiff
stiff = torch.broadcast_to(stiff, (len(x), 3, 3))
stiff_bc = torch.broadcast_to(stiff_bc, (len(Boundary), 3, 3))

# Define optimizer and learning rate scheduler
params = list(stress_net.parameters()) + list(disp_net.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

# Record training loss for plotting
loss_history = []

for epoch in range(iterations):
    scheduler.step()
    optimizer.zero_grad()

    # To compute stress from stress net
    sigma = stress_net(x)
    disp = disp_net(x)

    # displacement in x direction
    u = disp[:, 0]
    # displacement in y direction
    v = disp[:, 1]

    # Compute spatial derivatives via automatic differentiation
    dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # Define strain
    e_11 = dudx[:, 0].unsqueeze(1)
    e_22 = dvdx[:, 1].unsqueeze(1)
    e_12 = 0.5 * (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)
    e = torch.cat((e_11, e_22, e_12), dim=1).unsqueeze(2)

    # Augmented stress from constitutive law (Hooke's law)
    sig_aug = torch.bmm(stiff, e).squeeze(2)
    loss_cons = loss_func(sig_aug, sigma)

    # find displacement and stress at the boundaries
    disp_bc = disp_net(Boundary)
    sigma_bc = stress_net(Boundary)
    u_bc = disp_bc[:, 0]
    v_bc = disp_bc[:, 1]

    # Compute the strain and stresses at the boundary
    dudx_bc = torch.autograd.grad(u_bc, Boundary, grad_outputs=torch.ones_like(u_bc), create_graph=True)[0]
    dvdx_bc = torch.autograd.grad(v_bc, Boundary, grad_outputs=torch.ones_like(v_bc), create_graph=True)[0]
    e_11_bc = dudx_bc[:, 0].unsqueeze(1)
    e_22_bc = dvdx_bc[:, 1].unsqueeze(1)
    e_12_bc = 0.5 * (dudx_bc[:, 1] + dvdx_bc[:, 0]).unsqueeze(1)
    e_bc = torch.cat((e_11_bc, e_22_bc, e_12_bc), dim=1).unsqueeze(2)

    sig_aug_bc = torch.bmm(stiff_bc, e_bc).squeeze(2)
    loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

    # Equilibrium (PDE residuals)
    sig_11 = sigma[:, 0]
    sig_22 = sigma[:, 1]
    sig_12 = sigma[:, 2]

    dsig11dx = torch.autograd.grad(sig_11, x, grad_outputs=torch.ones_like(sig_11), create_graph=True)[0]
    dsig22dx = torch.autograd.grad(sig_22, x, grad_outputs=torch.ones_like(sig_22), create_graph=True)[0]
    dsig12dx = torch.autograd.grad(sig_12, x, grad_outputs=torch.ones_like(sig_12), create_graph=True)[0]

    eq_x1 = dsig11dx[:, 0] + dsig12dx[:, 1]
    eq_x2 = dsig12dx[:, 0] + dsig22dx[:, 1]

    # Zero body forces
    f_x1 = torch.zeros_like(eq_x1)
    f_x2 = torch.zeros_like(eq_x2)

    loss_eq1 = loss_func(eq_x1, f_x1)
    loss_eq2 = loss_func(eq_x2, f_x2)

    # Boundary conditions
    tau_R = 0.1
    tau_T = 0.0

    # Evaluate on specific boundaries
    u_L = disp_net(L_boundary)  # Left boundary: u = 0 (symmetry)
    u_B = disp_net(B_boundary)  # Bottom boundary: v = 0 (symmetry)
    sig_R = stress_net(R_boundary)  # Right boundary: traction in x-direction
    sig_T = stress_net(T_boundary)  # Top boundary: traction in y-direction
    sig_C = stress_net(C_boundary)  # Circular (hole) boundary: traction free

    # Essential (Dirichlet) boundary conditions
    loss_BC_L = loss_func(u_L[:, 0], torch.zeros_like(u_L[:, 0]))  # u=0 on left boundary
    loss_BC_B = loss_func(u_B[:, 1], torch.zeros_like(u_B[:, 1]))  # v=0 on bottom boundary

    # Neumann (traction) boundary conditions
    loss_BC_R = loss_func(sig_R[:, 0], tau_R * torch.ones_like(sig_R[:, 0])) + \
                loss_func(sig_R[:, 2], torch.zeros_like(sig_R[:, 2]))
    loss_BC_T = loss_func(sig_T[:, 1], tau_T * torch.ones_like(sig_T[:, 1])) + \
                loss_func(sig_T[:, 2], torch.zeros_like(sig_T[:, 2]))
    # Traction free on the circular hole boundary
    loss_BC_C = loss_func(sig_C[:, 0] * C_boundary[:, 0] + sig_C[:, 2] * C_boundary[:, 1],
                          torch.zeros_like(sig_C[:, 0])) + \
                loss_func(sig_C[:, 2] * C_boundary[:, 0] + sig_C[:, 1] * C_boundary[:, 1],
                          torch.zeros_like(sig_C[:, 0]))

    # Total loss (sum of all contributions)
    loss = loss_eq1 + loss_eq2 + loss_cons + loss_BC_L + loss_BC_B + loss_BC_R + loss_BC_T + loss_BC_C + loss_cons_bc

    # Uncomment the following block to include measurement data (Part e)
    #x_fix = x_full[rand_index, :]
    #u_fix = disp_net(x_fix)
    #loss_fix = loss_func(u_fix, disp_fix)
    #loss = loss + 100 * loss_fix

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Iteration {epoch}, Loss: {loss.item()}")

# Plot the training loss history
plt.figure()
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Error against Training Epochs")
plt.grid(True)
plt.show()

# Plot the tensile stress field 
# Prepare the stiffness matrix for all points in x_full
stiff_full = E / (1 - mu ** 2) * torch.tensor([[1, mu, 0],
                                               [mu, 1, 0],
                                               [0, 0, (1 - mu) / 2]], dtype=torch.float64)
stiff_full = stiff_full.unsqueeze(0)
stiff_full = torch.broadcast_to(stiff_full, (len(x_full), 3, 3))

u_full = disp_net(x_full)
stress_full = stress_net(x_full)

xx = x_full[:, 0].detach().numpy()
yy = x_full[:, 1].detach().numpy()
# σ₁₁ is the first component of the stress output
sig11 = stress_full[:, 0].detach().numpy()

connect = (t_connect - 1).detach().numpy()
triang = mtri.Triangulation(xx, yy, connect)

plt.figure()
plt.tricontourf(triang, sig11, levels=50, cmap='jet')
plt.title("Tensile Stress Field σ₁₁")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
