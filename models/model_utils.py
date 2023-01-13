import torch
from keras import backend


def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


def false_positive(self, out_tensor):
    mask = out_tensor.ge(0.5)
    mask = mask.view(len(out_tensor))
    binary_tensor = torch.cuda.FloatTensor(len(out_tensor)).fill_(0)
    binary_tensor.masked_fill_(mask, 1.)
    return binary_tensor.sum()


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake dataloader
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake dataloader
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
