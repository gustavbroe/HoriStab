'''
Visualizes correlations in the dataset by plotting:

- Pearson correlation coefficients between each attribute 
  and the target variable.

- The full attribute-to-attribute 
  correlation matrix as a heatmap.

This helps identify relationships and dependencies among 
features and the target.

Written by Gustav Broe Hansen - s193855
Created on Sunday March 26 12:45 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

from PlotSettings import *
import DataExtraction as DE

# Type of dataset
Type = 'Bar'
X, y, AttName, AttSymb, AttUnit, N, M, _ = DE.ImportData(Type)



# %%
# ------------ CORRELATIONS - ATTRIBUTES x TARGET ------------

# Aspect specific to the structure type
# - Axis and color nomalization limits.
# - Relative height of the figure.
if Type == 'Bar':
    lim = 1.0
    rh = 0.2
elif Type == 'Frame':
    lim = 0.4
    rh = 0.6

# Compute Pearson correlation between each feature and y
corr_with_target = np.array(
    [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
)

# Print correlations with attribute names
for name, corr in zip(AttName[:-1], corr_with_target):
    print(f"{name}: {corr:.2f}")

# Initlialize figure
fig, ax = plt.subplots(1, 1, figsize=(21.0*cm*0.8, 29.7*cm*0.8*rh), 
                       layout='constrained')
fig.canvas.manager.set_window_title("Correlation with target")

# Define normalization for the colormap 
norm = mpl.colors.Normalize(vmin=-lim, vmax=lim)
cmap = plt.get_cmap(MPCM_nea)

# Plotting using horizontal bar chart
bars = ax.barh(
    range(len(corr_with_target)),
    [abs(c) for c in corr_with_target],  # All bars to the left of 0
    color=cmap(norm(corr_with_target))
)

# Adding data labels to the bars
for i, v in enumerate(corr_with_target):
    ax.text(abs(v) + 3E-3, i, f"{v:.2f}", 
            va='center', ha='left', color='black', fontsize=10)

# Beautifing the plot
ax.set_yticks(range(len(corr_with_target)))
ax.set_yticklabels(AttName[:-1], fontsize=10)
ax.set_xlabel(f'Correlation with {AttName[-1]}', fontsize=12)
ax.set_ylabel('Attributes', fontsize=12)
ax.set_xlim([0, lim])
ax.grid(axis='x')
ax.set(frame_on=False)

# Reverting the order back to original
ax.invert_yaxis()

# Printing
plt.savefig('Figures/'+Type+'/Correlation.pdf', format='pdf', 
            bbox_inches='tight')



# %%
# ------------ CORRELATIONS - ATTRIBUTES x ATTRIBUTES ------------

# Compute correlation matrix between all attributes (excluding target)
corr_matrix = np.corrcoef(X, rowvar=False)

# Initialize figure for correlation matrix
fig, ax = plt.subplots(1, 1, figsize=(21.0*cm*0.8, 29.7*cm*0.8*0.5))
fig.canvas.manager.set_window_title("Attribute correlation matrix")

# Plot the correlation matrix as a heatmap
im = ax.imshow(corr_matrix, cmap=MPCM_nea, vmin=-1, vmax=1)

# Axis labels
AN = AttName[:-1] 
AS = [f'${a}$' for a in AttSymb[:-1]]

# Set axis ticks and labels
ax.set_xticks(np.arange(len(AS)))
ax.set_yticks(np.arange(len(AN)))
ax.set_xticklabels(AS, rotation=90, ha='center', fontsize=10)
ax.set_yticklabels(AN, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, 
                    ticks=[-1, -0.5, 0, 0.5, 1])
cbar.set_label('Correlation between attributes', fontsize=12,
               rotation=270, labelpad=15)

# Annotate correlation values
if len(AN) < 7:
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                    ha='center', va='center', color='black', 
                    fontsize=7)

# Fixing layout and printing
fig.tight_layout()
plt.savefig(f'Figures/{Type}/CorrelationMatrix.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()