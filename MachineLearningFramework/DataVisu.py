'''
Graphical representaion of the data set, which provides
insight into possible correlation between attributes.

Written by Gustav Broe Hansen - s193855
Created on Sunday March 26 12:45 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

from PlotSettings import *
import DataExtraction as DE

# Type of dataset
# (Junk attributes should not be removed)
Type = 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Printing dataset statistics to CLI 
DE.DispStats(np.insert(X, X.shape[1], y, axis=1), AttName)

# Linear regression
def LinReg(XTr, yTr):

    # Precomputing terms
    Xty = XTr.T @ yTr
    XtX = XTr.T @ XTr

    # Finding the optimal weights
    try:
        w = np.linalg.solve(XtX, Xty).squeeze()
    except np.linalg.LinAlgError:
        # Fall back to least squares solution if singular
        w = np.linalg.lstsq(XTr, yTr, rcond=None)[0].squeeze()

    return w



# %%
# -------------- ATTRIBUTES X OUTPUT  --------------

# Aspect specific to the structure type
# - Number of plots in width and height.
# - Attributes to plot.
# - Relative height of the figure.
if Type == 'Bar':
    NoX, NoY = 2, 2
    ap = np.arange(X.shape[1])
    rh = 0.5
    ylim = 80       # np.max(y)
elif Type == 'Frame':
    NoX, NoY = 3, 4
    ap = [0, 1, 2, 
          4, 5, 6, 
          8, 9, 10,
          16, 13, 17]
    rh = 0.9
    ylim = 9

# Total number of plots
NoP = NoX * NoY

# Check if the number of attributes matches the number of plots
if len(ap) != NoP:
    raise ValueError(f"Number of attributes in 'ap' ({len(ap)}) does "
                     f'not match number of subplots ({NoP}).')

# Initializing figure
fig, ax = plt.subplots(NoY, NoX, figsize=(21.0*cm*0.8,
    29.7*cm*0.8*rh), layout='constrained')

# Naming and positioning of figure window
fig.canvas.manager.set_window_title("Attributes x Output")
# fig.canvas.manager.window.wm_geometry("+100+100")

# Define normalization for the colormap 
norm = mpl.colors.Normalize(vmin=0, vmax=ylim)

# Flatten axes
ax = ax.flatten()

# Looping over each subplot in the figure
for j, idx in enumerate(ap):
    # Current axis
    cax = ax[j]

    # Plotting each attribute as a function of y
    sp = cax.scatter(X[:,idx], y, s=2, alpha=1.0, edgecolors='none',
                     c=y, cmap=MPCM, norm=norm,
                     zorder=-5, rasterized=True)
    
    # Only rasterizing the scatter plot
    cax.set_rasterization_zorder(0)

    # Linear regression line with offset and normalized slope
    xj = X[:, idx]
    w = LinReg(np.c_[np.ones(N), xj], y)
    x_fit = np.linspace(xj.min(), xj.max(), 100)
    y_fit = w[0] + w[1] * x_fit
    cax.plot(x_fit, y_fit, color=Col['G'], linewidth=2, zorder=10, ls='-')

    # Normalized slope (using standardized x)
    slope_n = LinReg(np.c_[np.ones(N), X_tilde[:, idx]], y)[1]
    print(f"{AttName[idx]} - {w[1]:.3f} ({slope_n:.3f})")

    # Add slope annotation in the top right corner
    cax.annotate(f'{w[1]:.2f} ({slope_n:.2f})',
                 xy=(0.96, 0.96), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                #  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                 )

    # Adding grid lines
    # cax.grid('true')

    # Setting y-axis limits
    cax.set_ylim(0, ylim)
    
    # Adding x-label
    cax.set_xlabel(FormatLabel(AttSymb[idx], AttUnit[idx]))

    # Alternatively, with the attribute names
    # an = AttName[idx]
    # sep = ',\n' if len(an) > 12 else ', '
    # cax.set_xlabel(fr'{an}{sep}${AttSymb[idx]}' 
    #                 fr'\: [{AttUnit[idx]}]$')

    # Applying specific x-ticks
    SetTicks(cax, AttSymb, idx, Type, 'x')


# Hide any unused subplots in the attribute grid
for i in range(len(ap), len(ax)):
    fig.delaxes(ax[i])

# Adding a global y-label
tt = fr'{AttName[-1]}, ${AttSymb[-1]} \: \left[{AttUnit[-1]}\right]$'
fig.supylabel(tt)   # 'Each attribute against ' + 

# Fixing the layout, ensuring that labels and title are visible
fig.tight_layout(w_pad=1.4, h_pad=1.4)

# Printing and displaying
plt.savefig(f'Figures/{Type}/{Type}VisOutput.pdf', format='pdf', 
            bbox_inches='tight', dpi=300)



# %%
# ------------ ATTRIBUTES X ATTRIBUTES  ------------

# Type specific settings
# - Pairs of attributes to plot.
# - Normalization limits for the colormap.
if Type == 'Bar':
    # All unique pairs (i, j) with i < j
    NoP = X.shape[1]
    selected_pairs = [(i, j) for i in range(NoP) 
                             for j in range(i+1, NoP)]
    vmin, vmax = 0, 80
elif Type == 'Frame':
    selected_pairs = [
        (8, 9), (4, 5),
        (4, 8), (13, 17),
        (0, 15), (0, 16),
    ]
    vmin, vmax = 0, 9

# Total number of plots and in each directions
NoP = len(selected_pairs)
NoPx, NoPy = int(np.ceil(NoP/2)), 2

# Initializing figure
fig, ax = plt.subplots(NoPx, NoPy, figsize=(21.0*cm*0.8, 
        29.7*cm*0.8*0.7), layout='constrained')
fig.canvas.manager.set_window_title("Attributes x Attributes")
# fig.canvas.manager.window.wm_geometry("+1000+100")

# Flatten axes
ax = ax.flatten()

# Plotting each attribute pair
for j, (px, py) in enumerate(selected_pairs):
    # Current axis
    cax = ax[j]

    # Normalization rule    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # Plotting each attribute as a function of y
    if Type == 'Bar':
        sp = cax.scatter(X[:,px], X[:,py], s=y*4, lw=0.3, marker='.',
                        c=y, cmap=MPCM, norm=norm,
                        zorder=-5, rasterized=True)
        
        # Only rasterizing the scatter plot
        cax.set_rasterization_zorder(0)
        
        # Changing from filled dots to just outline
        sp.set_facecolor('none')
        sp.set_edgecolor(MPCM(norm(y)))
    elif Type == 'Frame':
        sp = cax.scatter(X[:,px], X[:,py], s=y*14, alpha=0.4, marker='.',
                        c=y, cmap=MPCM, norm=norm, edgecolor='none',
                        zorder=-5, rasterized=True)

        # Add linear regression plane for the attribute pair
        X_pair = np.column_stack((np.ones(N), X[:, [px, py]]))
        w = LinReg(X_pair, y)
        x1_fit = np.linspace(X[:, px].min(), X[:, px].max(), 50)
        x2_fit = np.linspace(X[:, py].min(), X[:, py].max(), 50)
        X1_grid, X2_grid = np.meshgrid(x1_fit, x2_fit)
        y_fit = w[0] + w[1]*X1_grid + w[2]*X2_grid

        # Plot regression lines as contours
        cax.contour(X1_grid, X2_grid, y_fit, levels=3, 
                    colors=Col['SecG'], linewidths=1.2, zorder=10)
    
    # Increase axis margins for better visibility
    cax.use_sticky_edges = False
    cax.margins(x=0.1, y=0.1)
    
    # Adding grid lines
    # cax.grid('true')  

    # Applying specific x-ticks
    SetTicks(cax, AttSymb, px, Type, 'x')
    # SetTicks(cax, AttSymb, py, Type, 'y')
        
    # Adding x- and y-labels
    cax.set_xlabel(FormatLabel(AttSymb[px], AttUnit[px]))
    cax.set_ylabel(FormatLabel(AttSymb[py], AttUnit[py]))

# Hide any unused subplots
for i in range(len(selected_pairs), NoPx*NoPy):
    fig.delaxes(ax.flatten()[i])

# Fixing the layout, ensuring that labels and title are visible
fig.tight_layout(w_pad=1.0, h_pad=1.8)

# Makes space for the colorbar
# (This does not effect the printed version)
fig.subplots_adjust(top=0.93)

# Add colorbar to the bottom
cbar = fig.colorbar(
    sp, ax=ax.ravel().tolist(),
    orientation='horizontal',
    aspect=50, fraction=0.03,
    pad=0.10, location='top'
)
cbar.ax.xaxis.set_ticks_position("bottom")
cbar.ax.xaxis.set_label_position("top")
# cbar.ax.xaxis.set_ticks(np.arange(15, np.ceil(max(y))+1, 15))
cbar.set_label(fr'{AttName[-1]}, '
               fr'${AttSymb[-1]} \: '
               fr'\left[{AttUnit[-1]}\right]$'
               , labelpad=10.5, loc='center')

# Printing and displaying
plt.savefig(f'Figures/{Type}/{Type}VisInput.pdf', format='pdf', 
            bbox_inches='tight', dpi=300)

# Displaying figure
plt.show()