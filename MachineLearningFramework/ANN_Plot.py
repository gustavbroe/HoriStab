'''
Two functions for visualizing a neural network,
see the description of the functions for more details.

Written by Gustav Broe Hansen - s193855
Created on Thrusday April 17 13:58 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

import numpy as np      # Easy mapping using np.array and more
import torch            # Loading models and extracting props.
import matplotlib.pyplot as plt     # Generating figures
from PlotSettings import Col        # Plotting colors

# Plotting functions for visualizing ANNs
from matplotlib.lines import Line2D
from matplotlib.patches import Circle



# %% 
# ----------------- FUNCTIONS -----------------

def ExtractModelParam(model):
    """
    Extracts weights, biases, and activation function strings
    from a PyTorch Sequential model.

    Args:
        model (torch.nn.Sequential): The PyTorch model.

    Returns:
        tuple: (weights, biases, transfer_functions)
               - weights (list): List of weight numpy arrays (transposed).
               - biases (list): List of bias numpy arrays.
               - transfer_functions (list): List of string representations
                 of activation functions.
    """

    weights = []
    biases = []
    transfer_functions = []
    # Default if linear is last layer or followed by non-activation
    possible_activation = 'Identity'

    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            # Append weight and bias from Linear layer
            # PyTorch Linear weight shape: (features_out, features_in).
            # We transpose to (features_in, features_out) for drawing.
            weights.append(layer.weight.data.numpy().T)
            biases.append(layer.bias.data.numpy())

            # Check *next* layer for activation function
            if i + 1 < len(model):
                next_layer = model[i+1]
                # Add more activation types if needed
                if isinstance(next_layer,
                              (torch.nn.ReLU, torch.nn.Sigmoid,
                               torch.nn.Tanh, torch.nn.Softmax)):
                    possible_activation = str(next_layer)
                else:
                    # e.g., followed by Dropout or another Linear
                    possible_activation = 'Identity'
            else:
                 # Last layer was Linear
                 possible_activation = 'Identity'

            # Adding to list of strings
            transfer_functions.append(possible_activation)

    return weights, biases, transfer_functions


def DrawANN(
    weights, biases, tf_names, attribute_names=None,
    figsize=(24, 16), fontsizes=(8, 6), weight_threshold=0.5
):
    """
    Visualizes a neural network from input to output 
    with given weights.
    It is a modified version of the 'draw_neural_net'
    function from 'visualize_nn.py' in the 'dtuimldmtools' 
    package, created by DTU compute.

    Args:
        weights (list): Numpy arrays, weights for each layer
                        (shape: features_in x features_out).
        biases (list): Numpy arrays, bias for each layer.
        tf_names (list): Strings describing the transfer function
                         *after* each layer's linear transformation.
        attribute_names (list, optional): Names for input nodes.
                                           Defaults to X_1, X_2,...
        figsize (tuple, optional): Figure size. Defaults to (24, 16).
        fontsizes (tuple, optional): Font sizes for node labels
                                     and weight labels.
        weight_threshold (float, optional): Min absolute weight value
                                            to display text label.
    """

    # Layout and Spacing
    LeftMargin = 0.05
    RightMargin = 0.95
    BottomMargin = 0.1  # Increased bottom margin for text
    TopMargin = 0.9
    HSpacingFactor = 1.2 # Multiplier for horizontal spacing
    NodeSizeDivisor = 4.0 # Smaller divisor -> larger nodes

    # Line and Arrow Styles
    MinAlpha = 0.2
    MinLineWidth = 0.4
    MaxLineWidth = 3.5
    CircleColor = Col['BR']
    ArrowColor = Col['BR']
    ArrowLw = 1
    ArrowHeadWidth = 0.01
    ArrowHeadLength = 0.02
    InputArrowXOffset = 0.12
    InputArrowLengthFactor = 0.16 # Relative to h_spacing
    OutputArrowXOffset = 0.04
    OutputArrowLengthFactor = 0.16 # Relative to h_spacing

    # Text and Labels
    TextWeightOffsetFactor = 0.07 # Offset for weight text
    TextWeightYOffset = -0.009 # Offset perpendicular to the line
    TfTextYOffset = 0.04 # Offset below layer for transfer function
    InputLabelXOffset = 0.13
    OutputLabelXOffset = 0.14
    HiddenLabelXOffset = -0.019
    HiddenLabelYOffset = -0.028

    # Determine list of layer sizes, including input and output
    # Input layer size: weights[0].shape[0]
    # Hidden layer sizes: weights[i].shape[1] for i=0..N-2
    # Output layer size: weights[-1].shape[1]
    layer_sizes = [weights[0].shape[0]]
    layer_sizes += [w.shape[1] for w in weights]
    n_layers = len(layer_sizes)

    # Setup canvas
    # Use constrained_layout for auto adjustments
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.canvas.manager.set_window_title('ANN Structure')
    ax.axis('off')
    ax.set_aspect('equal') # Can cause issues with constrained_layout

    # Calculate spacing dynamically
    v_spacing = (TopMargin - BottomMargin) / float(max(layer_sizes))
    # Handle case of single layer (n_layers=1) -> division by zero
    h_denom = float(max(1, n_layers - 1))
    h_spacing = (RightMargin - LeftMargin) / h_denom * HSpacingFactor

    node_radius = v_spacing / NodeSizeDivisor

    # Determine normalization for edge width and alpha
    max_abs_w = 1.0 # Default
    if weights:
        valid_weights = [np.abs(w) for w in weights if w.size > 0]
        if valid_weights:
             max_abs_w = np.max([np.max(arr) for arr in valid_weights])
    if max_abs_w == 0: max_abs_w = 1.0 # Avoid division by zero

    # --- Draw Input Arrows ---
    layer_top_0 = (v_spacing * (layer_sizes[0] - 1) / 2.0
                   + (TopMargin + BottomMargin) / 2.0)
    for m in range(layer_sizes[0]):
        y = layer_top_0 - m * v_spacing
        ax.arrow(
            LeftMargin - InputArrowXOffset, y, 
            h_spacing*InputArrowLengthFactor, 0,
            lw=ArrowLw, head_width=ArrowHeadWidth,
            head_length=ArrowHeadLength, fc=ArrowColor, 
            ec=ArrowColor
        )

    # --- Draw Nodes and Labels ---
    # Store node centers [(x,y), (x,y), ...] for each layer
    node_positions = []
    for n, layer_size in enumerate(layer_sizes):
        layer_nodes = []
        layer_top = (v_spacing * (layer_size - 1) / 2.0
                     + (TopMargin + BottomMargin) / 2.0)
        x = n * h_spacing + LeftMargin

        # Add Bias Node (only for non-output layers)
        if n < n_layers - 1:
            y_bias = layer_top + v_spacing  # Bias on top
            bias_center = (x, y_bias)
            layer_nodes.append(bias_center)  # Insert first in list

            bias_circle = Circle(bias_center, node_radius,
                                 color='white', ec=Col['G'], zorder=4)
            ax.add_patch(bias_circle)
            ax.text(x - node_radius*2, y_bias, '1',
                    fontsize=fontsizes[0], ha='right', va='center')
        
        for m in range(layer_size):
            y = layer_top - m * v_spacing
            node_center = (x, y)
            layer_nodes.append(node_center)

            circle = Circle(node_center, node_radius,
                            color='w', ec=CircleColor, zorder=4)
            ax.add_patch(circle)

            # Add Labels
            if n == 0: # Input Layer
                has_names = attribute_names and m < len(attribute_names)
                label = str(attribute_names[m]) if has_names \
                        else rf'$X_{{{m+1}}}$'
                ax.text(x - InputLabelXOffset, y, label,
                        fontsize=fontsizes[0], ha='right', va='center')
            elif n == n_layers - 1: # Output Layer
                if attribute_names and len(attribute_names) >= layer_sizes[n]:
                    label = str(attribute_names[-(m + 1)])
                else:
                    label = rf'$y_{{{m+1}}}$'
                ax.text(x + OutputLabelXOffset, y, label,
                        fontsize=fontsizes[0], ha='left', va='center')
            else: # Hidden Layer
                label = rf'$H_{{{m+1},{n}}}$'
                ax.text(x + HiddenLabelXOffset, 
                        y + HiddenLabelYOffset, label,
                        fontsize=fontsizes[0], ha='left', va='center')

        node_positions.append(layer_nodes)

        # Add Layer Type and Dimensionality Text
        layer_type_text = ""
        if n == 0:
            layer_type_text = "Input layer"
        elif n == n_layers - 1:
            layer_type_text = "Output layer"
        else:
            layer_type_text = f"Hidden layer, $H_{{{n}}}$"

        dim_text = rf'$\in \mathbb{{R}}^{{{layer_size}}}$'
        text_y_pos = BottomMargin - TfTextYOffset + 0.02

        ax.text(x, text_y_pos, layer_type_text + dim_text,
            fontsize=fontsizes[0], ha='center', va='top')


        # Add Transfer Function Text (once per layer, after linear)
        if n > 0: # Below hidden/output layers
            tf_str = f'Act: {tf_names[n-1]}'
            ax.text(x, BottomMargin - TfTextYOffset - 0.01, tf_str,
                fontsize=fontsizes[0], ha='center', va='top', 
                style='italic')


    # --- Draw Edges and Weight Labels ---
    edge_items = zip(node_positions[:-1], node_positions[1:])
    for n, (nodes_a, nodes_b) in enumerate(edge_items):
        # Shape: (layer_size_a, layer_size_b)
        current_weights = weights[n]
        target_has_bias = (n < n_layers - 2) 

        # Adjust target nodes accordingly
        nodes_b_to_draw = nodes_b[1:] if target_has_bias else nodes_b

        for m, (xa, ya) in enumerate(nodes_a):
            for o, (xb, yb) in enumerate(nodes_b_to_draw):
                if m == 0:
                    weight = biases[n][o]
                else:
                    weight = current_weights[m - 1, o]

                # Line properties based on weight
                color = Col['R'] if weight > 0 else Col['B']
                abs_w_norm = np.abs(weight) / max_abs_w
                linewidth = (abs_w_norm * (MaxLineWidth - MinLineWidth)
                             + MinLineWidth)
                
                # Calculate alpha
                alpha = MinAlpha + (1 - MinAlpha) * abs_w_norm 

                # Draw edges behind nodes slightly (zorder=1)
                line = Line2D([xa, xb], [ya, yb], c=color,
                              linewidth=linewidth, zorder=1, alpha=alpha)
                ax.add_artist(line)

                # Add weight text if above threshold
                if abs(weight) > weight_threshold:
                    angle_rad = np.arctan2(yb - ya, xb - xa)
                    angle_deg = np.degrees(angle_rad)

                    # Position text near start of line, offset outward
                    jitter_scale = 0.03  # amount of randomness
                    jitter_amount = np.random.uniform(-jitter_scale, 
                                                      jitter_scale)
                    offset = (node_radius + TextWeightOffsetFactor 
                              + jitter_amount)
                    text_x = xa + offset * np.cos(angle_rad)
                    text_y = ya + offset * np.sin(angle_rad)

                    # Calculate perpendicular offset
                    perp_x_offset = TextWeightYOffset * np.sin(angle_rad)
                    perp_y_offset = -TextWeightYOffset * np.cos(angle_rad)

                    # Apply perpendicular offset
                    text_x += perp_x_offset
                    text_y += perp_y_offset

                    # Ensure text is on top (zorder=5)
                    ax.text(
                        text_x, text_y, f'{weight:.2f}',
                        rotation=angle_deg, rotation_mode='anchor',
                        fontsize=fontsizes[1], ha='center',
                        va='center', zorder=5
                    )

    # --- Draw Output Arrows ---
    last_layer_idx = n_layers - 1
    layer_top_last = (v_spacing * (layer_sizes[last_layer_idx] - 1) 
                      / 2.0 + (TopMargin + BottomMargin) / 2.0)
    x_last_node = last_layer_idx * h_spacing + LeftMargin

    for m in range(layer_sizes[last_layer_idx]):
        y = layer_top_last - m * v_spacing
        arrow_start_x = x_last_node + OutputArrowXOffset
        arrow_dx = h_spacing * OutputArrowLengthFactor
        ax.arrow(
            arrow_start_x, y, arrow_dx, 0,
            lw=ArrowLw, head_width=ArrowHeadWidth,
            head_length=ArrowHeadLength, fc=ArrowColor, ec=ArrowColor
        )