# dlg_original.py

import torch
import torch.nn.functional as F
from torch.autograd import grad

def deep_leakage_from_gradients(model, origin_grad, criterion=torch.nn.CrossEntropyLoss()):
    # Initialize dummy data and labels with random noise
    dummy_data = torch.randn(origin_grad[0].size(), requires_grad=True)
    dummy_label = torch.randn(origin_grad[1].size(), requires_grad=True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    for iters in range(300):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True)

            # Calculate gradient difference (L2 norm)
            grad_diff = sum(((dummy_grad - origin_grad) ** 2).sum()
                            for dummy_g, origin_g in zip(dummy_grad, origin_grad))

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    return dummy_data, dummy_label
