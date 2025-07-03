'''
Trains a single ANN using a specific number of hidden units in a 
artificial neural networks through 1-layer cross validation.

Error function: Sum-of-squares

Written by Gustav Broe Hansen - s193855
Created on Sunday April 8 14:34 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

from ANN_Train import TrainANN
from PlotSettings import *
import DataExtraction as DE

# Importing machine learning repositories (function and datasets) 
import torch
import sklearn.model_selection as ms
import pandas as pd

# Importing timer
import time

# Type of dataset
Type = 'Frame'  # 'Bar' or 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Printing dataset statistics to CLI 
DE.DispStats(np.insert(X, X.shape[1], y, axis=1), AttName)

# Using the normalized dataset
X = X_tilde
print(np.shape(X))


# --------------- ARTIFICIAL NEURAL NETWORKS ---------------

# Hyper parameter, number of hidden units
lampda = 14

# Ensuring the correct shape of y (N, 1) and not (N, )
y = y.reshape(np.shape(y)[0], 1)

# Converting dataset to PyTorch tensors
# (Data types are adjusted also, as weights are by default float32, 
# and numpy array is float64 (double))
X, y = torch.tensor(X).float(), torch.tensor(y).float()

# 1-level K-fold crossvalidation, outer- and inner fold respectivly
K1 = 9
CV1 = ms.KFold(K1, shuffle=True)

# Record for intermediate results
RecName = ['Fold', 'Testing error', 'Training error',
           'TeEr No features', 'TrEr No features']
Rec = np.empty([K1, len(RecName)])

# Total number of ANN to train
Nt = K1

# Starting counter
jt = 0

# Outer fold
for j1, (TrInd, TeInd) in enumerate(CV1.split(X, y)):

    # Initializing array for error of each inner loop and models
    # ErrVa = np.empty(K1)
    # ErrTr = np.copy(ErrVa)

    # Dataset is split into training and testing sets
    XTr, yTr = X[TrInd,:], y[TrInd]
    XTe, yTe = X[TeInd,:], y[TeInd]

    # Marking the starting time
    tic = time.perf_counter()
        
    # Provides updates on the current fold and total progress
    print(f'\nCurrent fold: {j1:<3.0f} | Progress: {jt/Nt*1E2:.1f}%')

    # Training a ANN, see function description
    ANN, ErrTrHist = TrainANN(XTr, yTr, NoHU=lampda, RCTol=1E-12, MaxIte=70000)

    # Predicting values using the trained ANN
    yEst = ANN(XTe)

    # Computing testing- and training error
    ErTe = (torch.sum((yEst - yTe)**2)/len(yTe)).item()
    ErTr = ErrTrHist[-1, 1]

    # Errors (sum-of-squares) when always guessing the mean of y
    # (meaning no features are used)
    ErTrNF = ((yTr-yTr.mean())**2).sum()/len(yTr)
    ErTeNF = ((yTe-yTe.mean())**2).sum()/len(yTe)

    # Recording data
    Rec[j1, :] = [j1, ErTe, ErTr, ErTeNF, ErTrNF]

    # Printing testing error
    print(f'\nTesting error, RMSE: {np.sqrt(ErTe):.4f}')
    
    # Counting number of trained models
    jt += 1

    # Printing the elapsed time sinse start
    toc = time.perf_counter()
    print(f'Training time: {(toc-tic):<6.1f} sec '
            f'| ETC: {((toc-tic)*(Nt - jt)/60):<5.1f} min |')
    print('-'*49)

    # Saving the optimal model
    torch.save(ANN, f'Models/Singels/NN_{Type}{lampda}_{j1}_100.pt')


# Computing generalization error for model
N_Te = np.array([len(TeInd) for _, TeInd in CV1.split(X, y)])
GenErr = np.sum(N_Te/N * Rec[:, RecName.index('Testing error')])

# Outputting results of the last fold
print('Estimated generalization error (RMSE): '
    f'{lampda:<2} - {np.sqrt(GenErr):.4f}')

# Printing mean performance
PerfDisp(Rec[:,4], Rec[:,3], Rec[:,2], Rec[:,1], f'ANN_{lampda}')

# # Saving historic loss of the last fold
# df = pd.DataFrame(ErrTrHist, columns=['Iter', 'Loss'])
# df.to_csv(f'Outputs/{Type}/LossHist/NoHU_{lampda}_S.csv', 
#         index=False, float_format="%.4e")

# # Saving results to a file
# df = pd.DataFrame(Rec, columns=RecName)
# df.to_csv(f'Outputs/{Type}/ANN_Rec_{lampda}.csv', 
#             index=False, float_format="%.6f")

# Plot performance, comparing estimator with real values
# for the last fold
PerfPlot(yTe, yEst, AttUnit)
plt.savefig(f'Figures/{Type}/Performance/PerfANN_{lampda}.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()