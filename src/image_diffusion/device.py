"""Device to be used for the training of the Class-conditional image generation using DiT with flow matching."""
import torch


def get_device(target='auto'):
    device = None
    # If we do not pass a particular string, default in the following order:
    # cuda -> mps -> cpu
    if target == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # If we do pass a particular device target,
    #   then we want this target to be present
    # If it is not, we want to return an error
    if target == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            message = f'Device: requested={target} not available!'
            raise RuntimeError(message)

    elif target == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            message = f'Device: requested={target} not available!'
            raise RuntimeError(message)

    elif target == 'cpu':
            device = torch.device('cpu')

    if device is not None:
        torch_version = torch.__version__
        message = f'Device requested={target}\t-> {device}\n'
        message += f'\ttorch={torch_version}'

        if device.type == 'cuda':
            cuda_version = torch.version.cuda
            message += f'\tcuda={cuda_version}'

        print(message)
        return device

    else:
        message = f'Invalid key! Got: {target}'
        raise ValueError(message)
