'''
Identifies the optimal number of hidden units in a 
artificial neural networks through 2-layer cross validation.

Error function: Mean-squared-error (MSE)

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
Type = 'Bar'  # 'Bar' or 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Using the normalized dataset
X = X_tilde



# --------------- ARTIFICIAL NEURAL NETWORKS ---------------

# Ensuring the correct shape of y (N, 1) and not (N, )
y = y.reshape(np.shape(y)[0], 1)

# Converting dataset to PyTorch tensors
# (Data types are adjusted also, as weights are by default float32, 
# and numpy array is float64 (double))
X, y = torch.tensor(X).float(), torch.tensor(y).float()

# Hyper parameter, number of hidden units
if Type == 'Bar':
    lampda = [2, 8, 10, 12, 14, 16, 20]
elif Type == 'Frame':
    lampda = [2, 8, 14, 18, 24, 28]

# 2-level K-fold crossvalidation, outer- and inner fold respectivly
K1, K2 = 10, 7
CV1, CV2 = ms.KFold(K1, shuffle=True), ms.KFold(K2, shuffle=True)

# Initializing array for optimal model and test errors
ResName = ['Optimal hyperparameter', 'RMSE', 'R-squared']
Res = np.empty([K1, len(ResName)])

# Record for intermediate results
RecName = ['Outer fold', 'Inner fold', 'Hyperparameter', 
           'Training error', 'Validation error', 
           'Weighted v. error']
Rec = np.empty([len(lampda)*K2*K1, len(RecName)])

# Total number of ANN to train
Nt = len(lampda)*K2*K1 + K1

# Starting counter
jt = 0

# Outer fold
for j1, (PaInd, TeInd) in enumerate(CV1.split(X, y)):

    # Dataset is split in a partion for validation of the 
    # model types and testing of the optimal model
    XPa, yPa = X[PaInd,:], y[PaInd]
    XTe, yTe = X[TeInd,:], y[TeInd]

    # Initializing array for error of each inner loop and models
    ErrVa = np.empty([len(lampda), K2])
    ErrTr = np.copy(ErrVa)

    # Inner fold
    for j2, (TrInd, VaInd) in enumerate(CV2.split(XPa, yPa)):

        # Marking the starting time
        tic = time.perf_counter()

        # Dataset is split into training and validation sets
        XTr, yTr = X[TrInd,:], y[TrInd]
        XVa, yVa = X[VaInd,:], y[VaInd]

        # Looping over hyperparameters
        for jn, NoHU in enumerate(lampda):
            
            # Provides updates on the current fold and total progress
            print(f'\nOuter fold: {j1:<3.0f} | Inner fold: {j2:<3.0f} '
                  f'| No hidden units: {NoHU:<3.0f} '
                  f'| Progress: {jt/Nt*1E2:.1f}%')
            
            # Adjusting stop condition for relative change for types
            RCTol = 1E-10 if Type == 'Frame' else 1E-6

            # Training a ANN, see function description
            ANN, ErrTrHist = TrainANN(XTr, yTr, NoHU=NoHU, 
                                      MaxIte=49000, RCTol=RCTol)

            # Predicting values using the trained ANN
            yEst = ANN(XVa)

            # Computing validation-, weighted, and training error
            EV = (torch.sum((yEst - yVa)**2)/len(yVa)).item()
            EVW = EV * len(yVa)/len(yPa)
            ET = ErrTrHist[-1, 1]

            # Recording data
            # (The counter is adjusted to ignore the optimal model
            # at the end of each outer fold)
            Rec[jt-j1, :] = [j1, j2, NoHU, ET, EV, EVW]
            
            # Counting number of trained models
            jt += 1

            # Saving historic loss of a hyperparameter
            if j1 == 0 and j2 == 0:
                df = pd.DataFrame(ErrTrHist, columns=['Iter', 'Loss'])
                df.to_csv(f'Outputs/{Type}/LossHist/NoHU_{NoHU}.csv', 
                        index=False, float_format="%.4e")


        # Printing the elapsed time sinse start
        toc = time.perf_counter()
        print('-'*49)
        print(f'\nTime elapsed: {(toc-tic):<6.1f} sec '
              f'| Est. remaining time: '
              f'{((toc-tic)*(Nt - jt)/60):<5.1f} min |')
            # f'{((toc-tic)*(K1*K2 - (j1*K2 + j2+1))/60):<5.1f} min |')


    # Computing generalization error for model types
    GenErr = [np.sum(Rec[(Rec[:,0]==j1) & (Rec[:,2]==l), -1]) 
              for l in lampda]

    # Outputting results of the current outer fold
    print('-'*49)
    print('Number of hidden units and associated estimated ')
    print(f'generalization error (RMSE) of outer fold, K1 = {j1:.0f}:')
    [print(f'{l:<2}' + ' - ' + f'{ge:.4f}') 
        for l, ge in zip(lampda, np.sqrt(GenErr))]
    print('-'*49)
        
    # Findint the optimal number of hidden units
    NoHUStar = lampda[np.argmin(GenErr)]
            
    # Provides updates on the current fold and total progress
    print(f'\nOuter fold: {j1:<3.0f} | Optimal No. HU {NoHUStar:<3.0f} '
            f'| Progress: {jt/Nt*1E2:.1f}%')

    # Training a model with the optimal hyperarameter
    ANN, _ = TrainANN(XPa, yPa, NoHU=NoHUStar)

    # Calculating the test error
    yEst = ANN(XTe)

    # Testing error
    ErrTe = (torch.sum((yEst - yTe)**2)/len(yTe)).item()

    # Baseline model and R-squared value
    ErrTeNF = (torch.sum((yTe.mean() - yTe)**2)/len(yTe)).item()
    R2 = (1 - ErrTe/ErrTeNF)

    # Saving results
    Res[j1, :] = [NoHUStar, np.sqrt(ErrTe), R2]

    # Saving the optimal model
    torch.save(ANN, f'Models/{Type}/NN_{j1}.pt')

    # Adding to the number of trained models
    jt += 1

# Saving results to a file
for (D, N, F) in zip([Res, Rec], [ResName, RecName], ['Res', 'Rec']):
    df = pd.DataFrame(D, columns=N)
    df.to_csv(f'Outputs/{Type}/ANN_{F}.csv', 
              index=False, float_format="%.6f")