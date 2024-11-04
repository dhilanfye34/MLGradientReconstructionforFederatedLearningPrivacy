# comb_methods.py

from dlg_original import deep_leakage_from_gradients
from inversefed.reconstruction_algorithms import improved_gradient_reconstruction

def combined_gradient_leakage(model, origin_grad, switch_iteration=100, num_iters=300, config=None):
    # Use the original DLG method for the initial iterations
    if switch_iteration > 0:
        dummy_data, dummy_label = deep_leakage_from_gradients(model, origin_grad)
    else:
        dummy_data = improved_gradient_reconstruction(model, origin_grad, config)

    # Optionally, add more logic to switch between the two methods at different points

    return dummy_data, dummy_label
