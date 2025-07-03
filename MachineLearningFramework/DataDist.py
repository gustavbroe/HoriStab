'''
Graphical representation of the distribution of 
selected attributes, which provides insight into 
the range and frequency of values.

Written by Gustav Broe Hansen - s193855
Created on Friday May 31 22:00 2025
'''

# %%
# ----------------- INITIALIZATION -----------------

from PlotSettings import *
import DataExtraction as DE

# Type of dataset
# (Junk attributes should not be removed)
Type = 'Frame'      # Only 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, _ = DE.ImportData(Type)

# Should not be used for 'Bar' dataset type
if Type == 'Bar':
    print("Error: This code is not tailored to the 'Bar' dataset type.")
    raise SystemExit

# Printing dataset statistics to CLI
print(np.shape(X))
DE.DispStats(np.insert(X, X.shape[1], y, axis=1), AttName)


# %%
# -------------- DISTRIBUTION OF ATTRIBUTES --------------

# Define the indices of the attributes to plot
attribute_indices = [1, 0, 16, 
                     17, 2, 4, 
                     8, 12, 13]

# Determine the number of attributes to plot
num_attributes = len(attribute_indices)

# Number of subplots in width and height
n_cols = 3
n_rows = int(np.ceil(num_attributes / n_cols))

# Initializing figure
fig, ax = plt.subplots(n_rows, n_cols, figsize=(21.0*cm*0.8, 
    29.7*cm*0.8*0.7), layout='constrained')

# Naming and positioning of figure window
fig.canvas.manager.set_window_title('Distribution of attributes')
# fig.canvas.manager.window.wm_geometry("+1000+100")

# Flatten axes for attribute subplots (excluding last row)
ax = ax.flatten()

# Iterate through the selected attribute indices and plot their distributions
for i, idx in enumerate(attribute_indices):

    # Current subplot axis
    cax = ax[i]

    # Visualize the distribution using histogram
    cax.hist(X[:, idx], bins=15, align='mid',
                alpha=1.0, color=Col['B'], edgecolor=None)

    # Adding x-label
    cax.set_xlabel(FormatLabel(AttSymb[idx], AttUnit[idx], 
                                AttName[idx], 12))
    # an = AttName[idx]
    # sep = ',\n' if len(an) > 12 else ', '
    # cax.set_xlabel(fr'{an}{sep}${AttSymb[idx]}' 
    #                fr'\: [{AttUnit[idx]}]$')

    # Set y-axis label
    if i % n_cols == 0:
        cax.set_ylabel('No. objects')

    # Ensuring reasonable number of y-ticks
    cax.yaxis.set_major_locator(
        ticker.MaxNLocator(3, integer=True))

    # Adding grid lines
    cax.grid('true')

    # Removing the margins
    cax.margins(x=0)

    # Applying specific x-ticks
    SetTicks(cax, AttSymb, idx, Type, 'x')

# Hide any unused subplots in the attribute grid
for i in range(num_attributes, len(ax)):
    fig.delaxes(ax[i])

# Fixing the layout, ensuring that labels and title are visible
# fig.tight_layout()

# Printing
plt.savefig('Figures/'+Type+'/DistAttributes.pdf', 
            format='pdf', bbox_inches='tight')



# -------------- DISTRIBUTION OF Y --------------

# Initializing figure
fig, ax = plt.subplots(1, 1, figsize=(21.0*cm*0.8, 
    29.7*cm*0.8*0.4), layout='constrained')

# Naming and positioning of figure window
fig.canvas.manager.set_window_title('Distribution of y')
# fig.canvas.manager.window.wm_geometry("+2000+100")

# Plot the histogram for the dependent variable
b = np.arange(0, 9+1, 1)
n, bins, patches = ax.hist(
    y, bins=b, align='mid', rwidth=0.96,
    color=Col['BR'], edgecolor='None', alpha=1.0, 
    )
ax.set_xlabel(FormatLabel(AttSymb[M], AttUnit[M], AttName[M]))
# ax.set_xlabel(fr'{AttName[M]} ${AttSymb[M]}' 
#                   fr'\: \left[{AttUnit[M]}\right]$')
ax.grid('true')
ax.margins(x=0)
ax.set_xticks(b)
ax.set_ylabel('No. objects')

# Limit y-axis
ymax = 2400
ax.set_ylim(top=ymax)

# Creating a white+solid image, 1 pixel wide and n_pixels tall
n_pixels = 21
rgba = np.ones((n_pixels, 1, 4))

# Adjusting the alpha channel to create a gradient
rgba[..., 3] = np.linspace(0.0, 1.0, n_pixels).reshape(-1, 1)

# Bottom of the gradient
ym = ymax - 200

# Show image
ax.imshow(rgba, aspect='auto', extent=[*ax.get_xlim(), ym, ymax],
          origin='lower', zorder=1)

# Annotate bars that exceed the y-limit
for count, left, right, patch in zip(n, bins[:-1], bins[1:], patches):
    if count > ymax:
        ax.annotate(
            f'{int(count)}',
            xy=((left + right) / 2, min(count, ymax)),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center', va='center', fontsize=12, 
            color=Col['BR'], fontstyle='normal', 
            fontweight='bold'
        )

# Fixing, printing and displaying
# fig.tight_layout()
fig.savefig('Figures/'+Type+'/DistDependent.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()