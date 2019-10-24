import torch
import torch.nn.functional as F
import math


def constant_annealing(model, iteration, max_iterations, start_tf=1.0):
    model.teacher_forcing = min(1, max(start_tf, 0))

def linear_annealing(model, iteration, max_iterations, start_tf=1.0):
    value = start_tf * (1 - iteration / max_iterations)
    model.teacher_forcing = min(1, max(value, 0))

def logarithmic_annealing(model, iteration, max_iterations, start_tf=1.0):
    value = start_tf * math.log(2) / math.log(iteration + 2)
    model.teacher_forcing = min(1, max(value, 0))

def exponentional_annealing(model, iteration, max_iterations, start_tf=1.0, gamma=0.99):
    value = gamma * model.teacher_forcing
    model.teacher_forcing = min(1, max(value, 0))

def fast_annealing(model, iteration, max_iterations, start_tf=1.0):
    model.teacher_forcing = min(1, max(start_tf / (iteration + 1), 0))

def sigmoid_annealing(model, iteration, max_iterations, start_tf=1.0, k=1):
    scale = (iteration / max_iterations) * (12) - 6
    value = start_tf * torch.sigmoid(torch.tensor(-scale)).item()
    model.teacher_forcing = min(1, max(value, 0))

def cosine_annealing(model, iteration, max_iterations, start_tf=1.0):
    scale = iteration / max_iterations
    value = start_tf * 0.5 * (1 + math.cos(scale * math.pi))
    model.teacher_forcing = min(1, max(value, 0))

def softplus_annealing(model, iteration, max_iterations, start_tf=1.0):
    max_value = math.log(math.e - 1)
    scale = (iteration / max_iterations) * (max_value + 5) - 5
    value = start_tf * (-F.softplus(torch.tensor(scale)).item() + 1.)
    model.teacher_forcing = min(1, max(value, 0))

def elu_annealing(model, iteration, max_iterations, start_tf=1.0):
    scale = (iteration / max_iterations) * (5) - 5
    value = start_tf * -F.elu(torch.tensor(scale)).item()
    model.teacher_forcing = min(1, max(value, 0))

def log_sigmoid_annealing(model, iteration, max_iterations, start_tf=1.0):
    max_value = math.log(math.e - 1)
    scale = (iteration / max_iterations) * (5 + max_value) - max_value
    value = start_tf * -torch.sigmoid(torch.tensor(scale)).log().item()
    model.teacher_forcing = min(1, max(value, 0))

def tanhshrink_annealing(model, iteration, max_iterations, start_tf=1.0):
    scale = (iteration / max_iterations) * 4 - 2
    value = start_tf * (-F.tanhshrink(torch.tensor(scale)).item() / (2 * 1.0360) + 0.5)
    model.teacher_forcing = min(1, max(value, 0))

def tanh_annealing(model, iteration, max_iterations, start_tf=1.0):
    scale = (iteration / max_iterations) * 12 - 6
    value = start_tf * (-torch.tanh(torch.tensor(scale)).item() * 0.5 + 0.5)
    model.teacher_forcing = min(1, max(value, 0))
