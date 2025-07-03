'''
Visulizing the main results from the ANN-analysis 
from 'ANN.py'-script.

Written by Gustav Broe Hansen - s193855
Created on Tuesday April 15 11:55 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

from PlotSettings import *
import ANN_Plot as PlotNN
import DataExtraction as DE

import numpy as np      # Easy mapping using np.array and more
import pandas as pd     # Reading csv files
import os               # Identifying hyperparameters
import torch            # Loading models and extracting props.

# Analytical expression for deflection
from BeamDeflection import TE

# Type of dataset
Type = 'Frame'  # 'Bar' or 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Recomputing statistical properties
mu = X.mean(axis=0)
sigma = np.std(X, axis=0)



# %% 
# ----------------- CONVERGENCE CHECK -----------------
# Looking a selection of hyperparameters for the first 
# outer- and inner fold.

# Reading the hyper parameter of avaliable loss histories
Folder = f'Outputs/{Type}/Losshist'
lampda = np.sort(np.array([int(f.split('_')[1].split('.')[0]) 
                            for f in os.listdir(Folder)]))

# Initializing figure for later adding loss histories
fig = plt.figure(figsize=(21.0*cm*0.8, 
        29.7*cm*0.8*0.4), layout='constrained')
fig.canvas.manager.set_window_title("Convergence check")

# Values of lampda to be plotted later
PltInd = np.linspace(0, len(lampda) - 1, 4, dtype=int)

# Starting counter
jp = 0

# Saving historic loss of a hyperparameter
for i in PltInd:

    # Reconstructing filepath
    File = Folder + '/NoHU_' + str(lampda[i]) + '.csv'

    # Reading csv-file
    ErrTrHist = pd.read_csv(File).values

    # Plotting the loss across iterations
    plt.plot(ErrTrHist[:,0]*1E-3, ErrTrHist[:,1], 
                label=lampda[i], linestyle='-', 
                linewidth=1.4, color=Col['SecG'][jp])

    # Beautifying the plot
    plt.legend(loc='best', fontsize=10, fancybox=False, 
        framealpha=1.0, draggable=True, 
        title='No. hidden units', title_fontsize=10)
    plt.xlabel('Number of iterations $[10^3]$')
    plt.ylabel(r'Mean squared training error, $E^\mathrm{train}$ '
                fr'$\left[ {AttUnit[-1]} \right]$')
    plt.grid('true')

    # Adding the number of plots
    jp += 1

# Printing plot
plt.savefig(f'Figures/{Type}/ConvergenceCheck.pdf', 
            format='pdf', bbox_inches='tight')



# %% 
# ------------ TRAINING- AND VALIDATION ERROR ------------
# Across all folds

# Loading data from csv-files
df = pd.read_csv(f'Outputs/{Type}/ANN_Rec.csv')
Rec = df.values
Name = df.columns.tolist()

# Initializing figure
fig = plt.figure(figsize=(21.0*cm*0.8, 
        29.7*cm*0.8*0.4), layout='constrained')
fig.canvas.manager.set_window_title('Training- and validation error')

# Excluding the first hyperparameter
lampda = np.delete(lampda, 0)

# Computing the average errors across folds for each lampda
extra = lambda c: np.transpose(np.array(
    [Rec[Rec[:,2]==l, c] for l in lampda]))

ErrTr = extra(Name.index('Training error'))
ErrVa = extra(Name.index('Validation error'))

# Shared x-labels
xL = r'Number of hidden units, $\lambda \: [1]$'

# Adding plots
plt.plot(range(0, len(lampda)), ErrVa.mean(axis=0), '.-', 
         color=Col['B'], label='Validation')
plt.plot(range(0, len(lampda)), ErrTr.mean(axis=0), '.-', 
         color=Col['G'], label='Training')
plt.grid(True)

# Alterning apearance
plt.xticks(range(0, len(lampda)), lampda)
plt.legend(loc='best', fontsize=10, fancybox=False, 
    framealpha=1.0, draggable=True)
plt.xlabel(xL)
plt.ylabel(fr'Mean squared error, $E(\lambda) ' 
            fr'\left[ {AttUnit[-1]} \right]$')

# Printing plot
plt.savefig(f'Figures/{Type}/TrainingValidationError.pdf', 
            format='pdf', bbox_inches='tight')

# Visualizes statistical properties of the testing error
BoxPlot(ErrTr, [str(l) for l in lampda], xL,
        r'Mean squared training error, $E^\mathrm{train} ' 
        fr'\left[ {AttUnit[-1]} \right]$')

# Printing plot
plt.savefig(f'Figures/{Type}/TrainingErrorStats.pdf', 
            format='pdf', bbox_inches='tight')



# %% 
# --------------- VISUALIZED NN STRUCTURE ---------------

# Loading the model
ModNum = 7
filepath = f'Models/{Type}/NN_{ModNum}.pt'
model = torch.load(filepath, weights_only=False)

# Extracting weights from the torhc model
weights, biases, tf = PlotNN.ExtractModelParam(model)

# Define figure size
FigW = 29.7*cm*0.9
FigH = 21.0*cm*0.8

# Drawing the ANN structure
PlotNN.DrawANN(weights, biases, tf, attribute_names=AttName, 
                figsize=(FigW, FigH), fontsizes=(10, 7),
                weight_threshold=2.1)

# Printing plot
plt.savefig(f'Figures/{Type}/ANNStructure{ModNum}.pdf', 
            format='pdf', bbox_inches='tight')



# %% 
# ------------- LOCKING A SINGLE PARAMETER -------------

# Varying one or more attributes for the Frame
def VarData(Type, VarInd, X, NoP=77):
    '''
    Create dataset by varying one parameter while keeping others constant.

    Parameters:
    - VarInd: index (0 to M) of the variable to vary
    - X: original data array (used to get min/max for varying parameter)
    - NoP: how many values to vary across

    Returns:
    - Xt: (NoP, M) matrix where one column varies
    - VarVal: (M,) non-normalized vector with the colm. that varies
    '''

    # Constants
    if Type == 'Frame':
        # 0=L - 1=H - 2=E_f - 3=I_cB - 4=I_cC - 5=I_bC - 6= I_bA - 
        # 7=C_cB - 8=C_C - 9=a_br - 10=k_br - 11=s_im - 12=alpha - 
        # 13=n_im
        ConstVal = [28, 6, 13, 50.44, 50.44, 129.35, 129.35, 30, 0, 0, 0, 0, 21, 0]
    elif Type == 'Bar':
        # 0 = L, 1 = C, 2 = E, 3 = h
        ConstVal = [4.0, 10.0, 9.0, 145.0]
        # ConstVal = [5.0, 60.0, 12.0, 95.0]

    # Generate variable values
    VarVal = np.linspace(X[:, VarInd].min(), X[:, VarInd].max(), NoP)

    # Insert into the correct position
    ConstVal[VarInd] = VarVal  

    # Create columns
    Xt = np.column_stack([
        np.full_like(VarVal, ConstVal[i]) if i != VarInd else VarVal
        for i in range(len(ConstVal))
    ])

    # Normalizing using original mean and std.dev.
    Xt = (Xt - mu) * (1 / sigma)

    return Xt, VarVal


# Translater function for determing the maximum deflection
# using the analytical expression
def yAnalytical(Xn, res=100):
    '''
    Computed the maximum deflection of an entry in dataset.
    Valid for a semi rigid bar, where parameters are indexed as:
        0 = L, 1 = C, 2 = E, 3 = h
    
    Parameters:
        Xn: ndarray of shape (N, 4) -- multiple input entries
        res: int -- number of points along the beam (resolution)
    
    Returns:
        yA: ndarray of shape (N,) -- max deflections
    '''
    a = Type
    # Known parameters
    b = 45E-3       # Width of cross section [m]
    q = 1E3         # Uniformly distributed load [N/m]

    # Number of entries in dataset
    N = Xn.shape[0]

    # Allocating memory for output vector
    yA = np.empty(N)
    
    # Converting the SI-units
    Xsi = Xn * [1., 1., 1E9, 1E-3]
    
    # Looping over each entry
    for j in range(N):

        # Position along the beam
        x = np.linspace(0, Xsi[j,0], res)

        # Second moment of area (rectangular)
        I_y = 1/12*b*Xsi[j,3]**3

        # Rotational stiffnesses [N*m/rad]
        K_A = Xsi[j,1]*(Xsi[j,2]*I_y)/Xsi[j,0]
        K_B = Xsi[j,1]*(Xsi[j,2]*I_y)/Xsi[j,0]

        # Shearing area (rectangular) [m^2]
        A_s = 5/6*b*Xsi[j,3]

        # Shear modulus (found in GPa converted to Pa)
        # [Table 2 - EN 384:2016]
        # G = np.interp(Xsi[j,2]*1E-9, E0mean, Gmean) *1E9
        G = Xsi[j,2]/16

        # Transverse deflection along the beam
        yx = TE(x, q, Xsi[j,0], Xsi[j,2], I_y, K_A, K_B, A_s, G)

        # Maximum deflection converted to transverse flexibility
        yA[j] = (yx.max()*1E3) / (q*1E-3)

    return yA

# Only for bar
if Type == 'Bar':

    # Initializing figure
    NoPx = NoPy = int(len(AttName)/2)
    fig, ax = plt.subplots(NoPx, NoPy, figsize=(21.0*cm*0.8, 
            29.7*cm*0.8*0.7), layout='constrained')
    fig.canvas.manager.set_window_title('Varying parameters')

    i = 0
    for px in range(0, NoPx):
        for py in range(0, NoPy):

            # Varied parameter
            AttName[i]

            # Creating a dataset where only one parameter varies
            Xt, VV = VarData(Type, i, X, NoP=77)

            # Reverting normalization of dataset
            Xn = Xt*sigma + np.ones((Xt.shape[0], 1))*mu

            # Allocating space for outputs
            yEst = np.empty((Xt.shape[0], 3))
            Prop = [None] * 3       # List of directories

            # Estimating using NLR
            # (Values from random_state=7297)
            Prop[0] = {'label': 'LR', 'color': Col['G'], 'linestyle': '-'}
            yEst[:,0] = (np.exp(np.ones(Xn.shape[0]))**14.102 
                        * Xn[:,0]**3.8361 * (Xn[:,1] + 0.2)**(-0.2550) 
                        * Xn[:,2]**(-0.9691) * Xn[:,3]**(-2.8489))

            # Estimating the transverse stiffness using the NN
            Prop[1] = {'label': 'ANN', 'color': Col['BR'], 'linestyle': '-'}
            yEst[:,1] = model(torch.tensor(Xt).float()).data.numpy().squeeze()

            # Computing the analytical solution
            Prop[2] = {'label': 'Analytical', 'color': Col['B'], 
                    'linestyle': '--'}
            yEst[:,2] = yAnalytical(Xn)

            # Plotting all the estimators against the varied par.
            for j in range(yEst.shape[1]):

                # Getting current axis
                cax = ax[px, py]

                # Plotting to current axis
                (obj,) = cax.plot(VV, yEst[:,j], lw=1.4, **Prop[j])

                # Adding x-label
                cax.set_xlabel(fr'{AttName[i]}, ${AttSymb[i]}'
                            fr' \: [{AttUnit[i]}]$')
                cax.grid(True)

            # Adding to counter
            i += 1

    # Fixing the layout, ensuring that labels and title are visible
    fig.tight_layout(w_pad=1.4, h_pad=2.8, )

    # Makes space for the colorbar
    # (This does not effect the printed version)
    fig.subplots_adjust(top=0.92, left=0.14)

    # Getting handels (small line the legend) and lines
    handle, labels = cax.get_legend_handles_labels()

    # Beautifying the plot
    fig.supylabel(fr'{AttName[-1]}, ${AttSymb[-1]} '
            fr'\: \left[ {AttUnit[-1]} \right] $', fontsize=14)
    fig.legend(handle, labels, loc='upper center', ncols=4, 
            fontsize=10, fancybox=False, framealpha=1.0, 
            draggable=True)

    # Printing plot
    plt.savefig(f'Figures/{Type}/ComparingModelTypes.pdf', 
                format='pdf', bbox_inches='tight')



    # %% 
    # ------------- COMPARING ANN MODELS -------------

    # Initializing figure
    NoPx = NoPy = int(len(AttName)/2)
    fig, ax = plt.subplots(NoPx, NoPy, figsize=(21.0*cm*0.8, 
            29.7*cm*0.8*0.7), layout='constrained')
    fig.canvas.manager.set_window_title('Comparing ANN models')

    # Possible ANN models
    Folder = f'Models/{Type}'
    os.listdir(Folder)
    Models = np.sort(np.array([int(f.split('_')[1].split('.')[0]) 
                                for f in os.listdir(Folder)]))

    # Number of models to plot
    # M = np.random.choice(Models, 4, replace=False)
    M = [0, 3, 4, 9]

    # Possible linestyles
    ls = ['-', '--', '-.', ':']

    i = 0
    for px in range(0, NoPx):
        for py in range(0, NoPy):

            # Creating a dataset where only one parameter varies
            Xt, VV = VarData(Type, i, X, NoP=77)

            # Getting current axis
            cax = ax[px, py]

            # Adding x-label and grids
            cax.set_xlabel(fr'{AttName[i]}, ${AttSymb[i]}'
                            fr' \: [{AttUnit[i]}]$')
            cax.grid(True)

            # Adding the different ANN models
            for j, m in enumerate(M):

                # Adding a title to legends (kind off)
                if px == NoPx-1 and py == NoPy-1 and j == 0:
                    cax.plot([], marker='', ls='',
                            label='Outer fold, $K_1$')[0]

                # Loading the model
                filepath = f'Models/{Type}/NN_{m}.pt'
                model = torch.load(filepath, weights_only=False)

                # Estimating the transverse stiffness using the NN
                yEst = model(torch.tensor(Xt).float()).data.numpy().squeeze()

                # Plotting to current axis
                cax.plot(VV, yEst, lw=1.4, label=m, color=Col['SecBR'][4-j], 
                        linestyle=ls[j])
                
            # Adding to counter
            i += 1

    # Fixing the layout, ensuring that labels and title are visible
    fig.tight_layout(w_pad=1.4, h_pad=2.8, )

    # Makes space for the colorbar
    # (This does not effect the printed version)
    fig.subplots_adjust(top=0.92, left=0.14)

    # Getting handels (small line the legend) and lines
    handle, labels = cax.get_legend_handles_labels()

    # Beautifying the plot
    fig.supylabel(fr'Estimated {AttName[-1].lower()}, ${AttSymb[-1]} '
            fr'\: \left[ {AttUnit[-1]} \right] $', fontsize=14)
    fig.legend(handle, labels, loc='upper center', ncols=5, 
            fontsize=10, fancybox=False, framealpha=1.0, 
            draggable=True, markerfirst=False)

    # Printing plot
    plt.savefig(f'Figures/{Type}/ComparingANNModels.pdf', 
                format='pdf', bbox_inches='tight')


# %% 
# ---------- CREATION OF SIMPLE VISUAL DESIGN TOOL ----------

# Aspect specific to the structure type
# - Number of plots in width and height.
# - Attributes to plot.
# - Relative height of the figure.
if Type == 'Bar':
    NoX, NoY = 2, 2
    ap = np.arange(X.shape[1])
    rh = 0.5
elif Type == 'Frame':
    NoX, NoY = 3, 2
    ap = [0, 1, 2, 
          3, 4, 5]
    rh = 0.4

# Models to compare
mod = [2, 3, 6, 7, 8]

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
fig.canvas.manager.set_window_title("Attributes x Prediction")
# fig.canvas.manager.window.wm_geometry("+100+100")

# Flatten axes
ax = ax.flatten()

# Looping over each subplot in the figure
for j, idx in enumerate(ap):
    
    # Creating a dataset where only one parameter varies
    Xt, VV = VarData(Type, idx, X, NoP=77)

    # Reverting normalization of dataset
    Xn = Xt*sigma + np.ones((Xt.shape[0], 1))*mu

    # Getting current axis
    cax = ax[j]

    # Define different line styles for each model
    linestyles = ['-', '--', '-.', ':']

    # Looping over models
    for i, m in enumerate(mod):

        # Optimal models from 2-level cross-validation
        # filepath = f'Models/{Type}/NN_{m}.pt'

        # Models from K-fold cross-validation
        filepath = f'Models/Singels/NN_{Type}14_{m}_100.pt'

        # Loading the model
        model = torch.load(filepath, weights_only=False)

        # Estimating the transverse stiffness using the NN
        yEst = model(torch.tensor(Xt).float()).data.numpy().squeeze()

        # Plotting to current axis with a different linestyle for each model
        (obj,) = cax.plot(VV, yEst, lw=1.4, c=Col['B'], label=m,
                          linestyle=linestyles[i % len(linestyles)])

    # Adding x-label
    if NoP < 6:
        t = fr'{AttName[idx]}, ${AttSymb[idx]} \: [{AttUnit[idx]}]$'
    elif NoP >= 6:
        t = fr'${AttSymb[idx]} \: [{AttUnit[idx]}]$'
    cax.set_xlabel(t)
    cax.grid(True)

# Fixing the layout, ensuring that labels and title are visible
fig.tight_layout(w_pad=1.4, h_pad=2.8, )

# Makes space for the colorbar
# (This does not effect the printed version)
fig.subplots_adjust(top=0.92, left=0.14)

# Hide any unused subplots in the attribute grid
for i in range(len(ap), len(ax)):
    fig.delaxes(ax[i])

# Adding a global y-label
tt = fr'{AttName[-1]}, ${AttSymb[-1]} \: \left[{AttUnit[-1]}\right]$'
fig.supylabel(tt)   # 'Each attribute against ' + 

# Getting handels (small line the legend) and lines
# handle, labels = cax.get_legend_handles_labels()
# fig.legend(handle, labels, loc='upper center', ncols=5, 
#         fontsize=10, fancybox=False, framealpha=1.0, 
#         draggable=True, markerfirst=False)

# Makes space for the legends
# fig.subplots_adjust(top=0.88, left=0.14)

# Fixing the layout, ensuring that labels and title are visible
fig.tight_layout(w_pad=1.4, h_pad=1.4)

# Printing and displaying
plt.savefig(f'Figures/{Type}/MainRes.pdf', format='pdf', 
            bbox_inches='tight', dpi=300)

# Displaying plots
plt.show()