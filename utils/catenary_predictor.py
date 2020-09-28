# Calculate the catenary of form
# b+a*(cosh((x-c)/a)-1)
# We will fit it using least squares method
import torch


def lsq_catenary(x, a, b, c):
    return b + a * (torch.cosh((x - c) / a) - 1)


# Calculate the loss
def lsq_loss(y, y_hat, error=lambda x: torch.norm(x)):
    return error(y - y_hat)


# The predictor class
# Given the points of x and y tries to predict the catenary that will fit to all of them
# Uses pytorch graidents
class CatenaryPredictor:
    def __init__(self, x, y, error=lambda x: torch.norm(x)):
        self.x = x
        self.y = y
        self.error_func = error
        # Init a,b,c arguments
        self.a = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        self.c = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

        # Init optimizer and scheduler
        self.optimizer = torch.optim.SGD([self.a, self.b, self.c], lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=1000, factor=0.80)

    # Fit the catenary
    def solve(self, iters=10000, verbosity=1000):
        # Used for early stopping
        prev_loss = 0  # Save previous loss
        patience = 0  # how many times losses have been the same

        for i in range(iters):
            self.optimizer.zero_grad()

            # Calculate new ys
            ys = lsq_catenary(self.x, self.a, self.b, self.c)
            # Calculate how different the ys are
            loss = lsq_loss(self.y, ys, self.error_func)
            if i % verbosity == 0:
                print(f'Epoch {i}. Loss {loss}')

            if loss == prev_loss:  # If losses were the same, increase counter
                patience += 1
            else:
                patience = 0  # Otherwise, reset

            if patience == 1000:  # If reached 1000 same epochs - stop
                break
            prev_loss = loss
            loss.backward()  # Update using gradient
            self.optimizer.step()
            self.scheduler.step(loss)
        return self.a, self.b, self.c, ys
