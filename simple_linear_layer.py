import torch
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from tqdm import tqdm

# think of 1 example case
# then think of batchification

def initialize_weight(fan_in, fan_out):
    weights = torch.empty(fan_in, fan_out)
    bound = torch.sqrt(torch.tensor(6.0 / fan_in))
    weights.uniform_(-bound, bound)
    return weights


class LinearLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # f = X.W_t + b
        # X -> [n, d_in]
        # W -> [d_out, d_in]
        # b -> [d_out]
        self.weight = initialize_weight(d_out, d_in)

        self.bias = torch.zeros((d_out))

    def forward(self, in_x):
        self.input = in_x
        return in_x @ self.weight.T + self.bias

    def calc_gradient(self, grad):
        # grad -> n x d_out
        # dl/dX -> (n,d_out) @ (d_out,d_in) -> (n, d_in) = grad @ W
        # dl/dW -> (d_out, d_in) -> (n, d_out)_t @ (n, d_in) = grad.T @ X 
        # dl/db -> d_out = sum(grad, dim=0)
        self.grad_input = grad @ self.weight
        self.grad_weight = grad.T @ self.input
        self.grad_bias = grad.sum(dim=0)
        return self.grad_input

    def backward_step(self, learning_rate=1e-5):
        self.weight -= learning_rate * self.grad_weight
        self.bias -= learning_rate * self.grad_bias

    def zero_grad(self):
        self.grad_weight, self.grad_bias, self.grad_input = None, None, None


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    # needs to be differentiable wrt x
    def forward(self, x):
        # ReLU = max(x,0)
        self.input = x        
        return torch.maximum(x, torch.tensor(0.0))

    def calc_gradient(self, grad):
        # df/dx = 1 where x > 0 else 0
        self.grad = grad.clone()
        self.grad[self.input < 0] = 0
        return self.grad

    def backward_step(self, learning_rate = 1e-5):
        pass

    def zero_grad(self):
        self.grad = None


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        self.input = input
        softmax = torch.exp(input)/torch.sum(torch.exp(input), dim=1)[:,None]
        log_softmax = torch.log(softmax)
        mask = torch.zeros_like(log_softmax)
        mask[torch.arange(len(mask)), target] = 1
        nll = - log_softmax * mask
        return nll.sum(dim=1).mean(dim=0)
    
    def calc_gradient(self):
        self.grad = (self.input + 1) / len(self.input)
        return self.grad

    def backward_step(self):
        pass

    def zero_grad(self):
        self.grad = None


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        # loss = sum((target - prediction)**2)/len(target)
        self.target = target
        self.input = x
        return torch.mean(torch.sum((target - x)**2, dim=-1),dim=0)

    def calc_gradient(self):
        self.grad = 2 * (self.target - self.input)/len(self.target)
        return self.grad

    def backward_step(self):
        pass

    def zero_grad(self):
        self.grad = None


class NeuralNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.module_list = nn.ModuleList()
        for idx in range(len(channels)-1):
            # self.module_list.append(LinearLayer(channels[idx], channels[idx + 1]))
            self.module_list.append(nn.Linear(channels[idx], channels[idx + 1]))
            self.module_list.append(ReLU())

    def forward(self, x):
        output = x
        for module in self.module_list:
            output = module(output)
        return output


    def zero_grad(self):
        for module in self.module_list:
            module.zero_grad()


    def backward_step(self, grad):
        for module in self.module_list[::-1]:
            grad = module.calc_gradient(grad)

        for module in self.module_list:
            module.backward_step()


if __name__ == "__main__":
    # Input and target
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize(mean=0.5, std=0.5)])
    mnist_train_dataset = torchvision.datasets.MNIST(root="data",train=True, transform=transformations, download=True)
    mnist_test_dataset = torchvision.datasets.MNIST(root="data",train=False, transform=transformations, download=True)

    train_dl = DataLoader(mnist_train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_dl = DataLoader(mnist_test_dataset, batch_size=8, shuffle=True, num_workers=0)

    # Initialize modules
    simple_linear_network = NeuralNetwork([784, 32, 10])
    optimizer = torch.optim.Adam(simple_linear_network.parameters())
    breakpoint()

    num_epochs = 100
    for _ in range(num_epochs):
        train_prog_bar = tqdm(train_dl)
        train_prog_bar.set_description("Training a simple linear layer from scratch on MNIST...")        

        for (img,label) in tqdm(train_dl):

            # zero grad
            # simple_linear_network.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            outputs = simple_linear_network.forward(img.reshape(img.shape[0], -1))
            # print(outputs)
            loss_criterion = CrossEntropy()
            loss = loss_criterion(outputs, label)

            train_prog_bar.set_description(f"Train Loss:{loss.item()}")

            # Backward pass
            # grad = loss_criterion.calc_gradient()
            # simple_linear_network.backward_step(grad)
            loss.backward()
            optimizer.step()


        test_prog_bar = tqdm(train_dl)
        test_prog_bar.set_description("Validating ...")        

        for img,label in tqdm(test_dl):
            outputs = simple_linear_network(img.reshape(img.shape[0], -1))
            loss = loss_criterion(outputs, label)
            test_prog_bar.set_description(f"Validation Loss:{loss.item()}")


