"""
Unify the appearanche of figures through the project.

Written by Gustav Broe Hansen - s193855
Created on Sunday March 18 08:23 2025
"""

# %%
# Initialization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.ticker as ticker

# Plotting functions for visualizing ANNs
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator
import torch

# Restore default settings
# plt.rcdefaults()

# Latex rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Font family and size
plt.rcParams["font.family"] = 'Segoe UI'
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["font.size"] = 10

# Line styles
plt.rcParams['lines.solid_joinstyle'] = 'round'
plt.rcParams['lines.solid_capstyle'] = 'round'
plt.rcParams['lines.dash_capstyle'] = 'butt'

# Always plot grid lines underneath data
plt.rcParams['axes.axisbelow'] = True

# Primary colors
Col = {}
Col['B']  = '#0000ff'
Col['R']  = '#c1170a'
Col['BR'] = '#a3593a'
Col['G']  = '#058a5f'

# Scondary colors
Col['SecB']  = ['#1000ff', '#0000d8', '#0000b2','#000065']
Col['SecR']  = ['#ff1e0d', '#e71b0b', '#c1170a','#740d06']
Col['SecBR'] = ['#ffb65a', '#ffa05a', '#bc6642','#a3593a','#562f1e']
Col['SecG']  = ['#04d491', '#03ae77', '#03885d','#036344','#023d2a']


# Defining a custom colormap with 3 colors from HEX and positions
colors = [Col['B'], Col['BR'], Col['R']] 
positions = [0, 0.5, 1]
MPCM = cl.LinearSegmentedColormap.from_list('MPCM', 
    list(zip(positions, colors)))

colors = [Col['B'], 'white', Col['R']] 
positions = [0, 0.5, 1]
MPCM_nea = cl.LinearSegmentedColormap.from_list('MPCM', 
    list(zip(positions, colors)))

# Unit conversion, cm to inch
cm = 1/2.54

# Function for adjusting ticks to preset
def SetTicks(ax, AttSymb, idx, Type, axis):
    # Shortening the current attribute symbol
    # (and removing latex formatting)
    cas = AttSymb[idx].replace(r'\mathrm{', '').replace('}}', '}', 1)

    # Adjusting ticks of each/specific attributes
    presets = {
        'Frame': {
            'H': (2, 7, (7-2)/1+1),
            'L': (7, 63, (63-7)/14+1),
            r'\alpha': (0, 25, (25-0)/5+1),
            'E_{f}': (10.4, 13.5, 4),
            'b': (115, 215, (215-115)/50+1),
            'n_{im}': (0, 4, (4-0)/1+1),
            'I_{c,B}': (3, 1125, 4),
            'E_{br}': (8.4, 210, 2),
            'C_': (0, 40, (40-0)/10+1),
        },
        'Bar': {
            'L': (3, 8, (8-3)/1+1),
            'E_{0,mean}': (7, 15, 4),
            'C_{A/B}': (0, 100, (100-0)/20+1),
            'h': (70, 220, (220-70)/25+1),
        }
    }

    # Set ticks on the specified axis
    ticks = None
    if Type in presets:
        params = presets[Type]
        if cas in params:
            s, e, n = params[cas]
            ticks = np.linspace(s, e, int(n))
        elif 'C_' in cas:
            s, e, n = params['C_']
            ticks = np.linspace(s, e, int(n))

    # Set ticks or use default locator
    if ticks is not None:
        (ax.set_xticks if axis == 'x' else ax.set_yticks)(ticks)
    else:
        locator = ticker.MaxNLocator(nbins=5, min_n_ticks=4, integer=True)
        (ax.xaxis.set_major_locator if axis == 'x' 
         else ax.yaxis.set_major_locator)(locator)

    # Adjust tick formatting for 'E_' attributes
    if 'E_' in cas:
        fmt = ticker.FormatStrFormatter('%.1f')
        (ax.xaxis.set_major_formatter if axis == 'x' 
         else ax.yaxis.set_major_formatter)(fmt)



# Function for formatting labels
def FormatLabel(symb, unit, name=None, limit=None):
    '''Formats a label for an axis, optionally 
    including the attribute name.
    Handles both regular and LaTeX 'frac' units.
    Inserts a line break after the name if it exceeds name_break_limit.'''
    label = ''
    if name is not None:
        if limit is not None and len(name) > limit:
            sep = ',\n'
        else:
            sep = ', '
        label += fr'{name}{sep}'
    if 'frac' in unit or '^' in unit:
        label += fr'${symb} \: \left[{unit}\right]$'
    else:
        label += fr'${symb} \: [{unit}]$'
    return label


# Function for displaying feature selection
def FSplot(bitmap, yTicks, xTicks, ax):
    r''' Plots a bitmap with "grid"-lines inbetween values 
    and adds specified ticks.
    (A modified version of 'bmplot' in the 'dtuimldmtools' package)
    "...\Lib\site-packages\dtuimldmtools\plots.py"
    '''

    # Map 1 to Col['G'] and 0 to 'white' using ListedColormap
    cmap = cl.ListedColormap(['white', Col['G']])
    plt.imshow(bitmap, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    
    # x- and y-axis tick marks
    ax.set_xticks(range(0, len(xTicks)), xTicks)
    ax.set_yticks(range(0, len(yTicks)), yTicks)

    # Adding horizontal and vertical lines 
    [ax.axvline(i - 0.5, c='k', lw=0.8) for i in range(0, len(xTicks))]
    [ax.axhline(i - 0.5, c='k', lw=0.8) for i in range(0, len(yTicks))]


def BoxTheme(bp, color):
    ''' Changes the color theme of the box plots '''

    # Forrmatting lines of each element
    for element in ['boxes', 'whiskers', 'fliers', 'means', 
                    'medians', 'caps']:
        plt.setp(bp[element], color=color, lw=1.4)
        if element == 'fliers':
            plt.setp(bp[element], markeredgecolor=color)

    # Formatting the filled region
    for patch in bp['boxes']:
        patch.set(facecolor='w')  


def BoxPlot(Xp, AttName, xLabel, yLabel, y=None):
    ''' Visualizes statistical properties of each attributes
    with the preferred color theme and style.'''

    # Initializing figure for final results
    cm = 1/2.54
    fig = plt.figure(figsize=(21.0*cm*0.8, 
            29.7*cm*0.8*0.4), layout='constrained')

    # Naming and positioning of figure window
    fig.canvas.manager.set_window_title("Box plot")

    # Plotting the selected features from each K-fold
    bp = plt.boxplot(Xp, patch_artist=True, widths=0.4)

    # Changing colors
    BoxTheme(bp, Col['B'])

    # Optionally adding the y-attribute 
    if y is not None:
        bp = plt.boxplot(y, positions=[Xp.shape[1] + 1], 
                         patch_artist=True, widths=0.4)
        BoxTheme(bp, Col['G'])

    plt.grid(axis='y')
    Att = ['\n'.join(A.split()) if len(A) else A for A in AttName]
    plt.xticks(range(1, len(Att)+1), Att, fontsize=10)
    plt.xlabel(xLabel, fontsize=12)
    plt.ylabel(yLabel)


def PerfPlot(yTe, yTeEst, AttUnit, color=Col['B']):
    ''' Visually compares the estimations with real values.'''

    # Convert tensors to numpy and flatten if necessary
    if isinstance(yTe, torch.Tensor):
        yTe = yTe.detach().numpy().flatten()
    if isinstance(yTeEst, torch.Tensor):
        yTeEst = yTeEst.detach().numpy().flatten()

    # Initializing figure for final results
    cm = 1/2.54
    fig = plt.figure(figsize=(21.0*cm*0.8, 
            29.7*cm*0.8*0.4), layout='constrained')

    # Naming and positioning of figure window
    fig.canvas.manager.set_window_title("Regression performance")
    # fig.canvas.manager.window.wm_geometry("+100+100")

    # Plotting the selected features from each K-fold
    plt.scatter(yTe, yTeEst, c=color, alpha=0.42, s=7, cmap=MPCM)
    plt.xlabel(fr'Actual value, $y \: \left[ {AttUnit[-1]} \right]$')
    plt.ylabel(r'Estimated value, $\hat{y} \: ' 
               fr'\left[ {AttUnit[-1]} \right]$')

    # Adding perfect line
    AxisLim = [np.min([yTe, yTeEst]) - 1, np.max([yTe, yTeEst]) + 1]
    plt.plot(AxisLim, AxisLim, "k--", lw=1.4)
    plt.ylim(AxisLim)
    plt.xlim(AxisLim)
    plt.grid(True)


def PerfDisp(ErTrNF, ErTeNF, ErTr, ErTe, Name):
    ''' Computes and displays the performance of a given 
    model and a baseline model given in mean error 
    and R-squared values. '''
    
    print('\n\nBaseline performance:')
    print(f'- Training error: {ErTrNF.mean():.3f}')
    print(f'- Test error:     {ErTeNF.mean():.3f}')

    print('Regression model: ('+Name+')')
    print(f'- Training error: {ErTr.mean():.3f}')
    print(f'- Test error:     {ErTe.mean():.3f}')
    print(f'- R^2 train:     {1 - ErTr.sum()/ErTrNF.sum():.3f}')
    print(f'- R^2 test:     {1 - ErTe.sum()/ErTeNF.sum():.3f}')