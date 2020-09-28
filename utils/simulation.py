import numpy as np
import torch


def catenary(x, a):
    return a * torch.cosh((x + 1) / a)  # Assume the form of catenary


def to_image(vector_x, camera):
    # Projects 3D vectors to 2D image coordinates given camera configuration
    Q = torch.Tensor(np.identity(3))  # Assume Q is identity
    t = torch.Tensor([0, 0, 100]).unsqueeze(1)  # Assume t is [0 0 100]

    concatenated = torch.cat((Q, t), 1)  # Get Q|t
    img = camera @ concatenated @ vector_x  # C[Q|T]x
    img = img / img[-1]  # divide by z to get coordinates
    return img


def add_noise(*vector_u):
    # Adds noise from N(0, 1) to all vectors in the input
    return list(map(lambda x: x + torch.normal(0, 1, (2,)), vector_u))


def calculate_error(u, u_hat, error_func=lambda x: torch.norm(x, 2)):
    # Calculates error function, given by the user (second norm by default)
    return error_func(u - u_hat)


def calculate_step(a, x_1, x_2, camera_matrix):
    # One step of the optimization
    y_1 = catenary(x_1, a)  # Calculate y1 and y2 values using assumed form of the catenary
    y_2 = catenary(x_2, a)

    # Stack values of xs and ys to get the 3D vector (z=0 as they are in the same plane)
    zero, one = torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32)
    vector_x1 = torch.stack((x_1, y_1, zero, one))
    vector_x2 = torch.stack((x_2, y_2, zero, one))

    # Project to the image space
    u1 = to_image(vector_x1, camera_matrix)[:2]
    u2 = to_image(vector_x2, camera_matrix)[:2]

    return u1, u2


# Class for simulation steps
class Simulation:
    # We need initial assumption of a, x1,x2 and camera matrix
    def __init__(self, a, x1, x2, camera_matrix):
        self.camera_matrix = camera_matrix
        # Calculate the target 'tilde u1' and 'tilde u2'
        self.perfect_u1, self.perfect_u2 = add_noise(*calculate_step(a, x1, x2, camera_matrix))

        # Trainables a, x1 and x2
        self.a = torch.tensor(1, dtype=torch.float32, requires_grad=True)
        self.x_1 = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        self.x_2 = torch.tensor(0, dtype=torch.float32, requires_grad=True)

        # Init optimizer and learning rate scheduler
        self.optimizer = torch.optim.SGD([self.a, self.x_1, self.x_2], lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.99)

    def calculate_step(self):
        # One step of the optimization
        y_1 = catenary(self.x_1, self.a)  # Calculate y1 and y2 values using assumed form of the catenary
        y_2 = catenary(self.x_2, self.a)

        # Stack values of xs and ys to get the 3D vector (z=0 as they are in the same plane)
        zero, one = torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32)
        vector_x1 = torch.stack((self.x_1, y_1, zero, one))
        vector_x2 = torch.stack((self.x_2, y_2, zero, one))

        # Project to the image space
        u1 = to_image(vector_x1, self.camera_matrix)[:2]
        u2 = to_image(vector_x2, self.camera_matrix)[:2]

        return u1, u2

    def simulate(self, iters=10000, verbosity=10000):
        # Simulation of convergence
        for i in range(iters):
            self.optimizer.zero_grad()

            # Get current u's
            new_u1, new_u2 = calculate_step(self.a, self.x_1, self.x_2, self.camera_matrix)

            # Calculate loss
            loss = (calculate_error(self.perfect_u1, new_u1) + calculate_error(self.perfect_u2, new_u2)) / 2
            if i % verbosity == 0:
                print(f'Epoch {i}. Loss {loss}')

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
        return self.a, self.x_1, self.x_2, new_u1, new_u2
