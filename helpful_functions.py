import torch


def get_device():
    if torch.mps.is_available():
        device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        print('Using MPS device:', device)
    elif torch.backends.cudnn.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using CUDA device:', device)
    else:
        device = torch.device('cpu')
        print('Using CPU device:', device)
    return device