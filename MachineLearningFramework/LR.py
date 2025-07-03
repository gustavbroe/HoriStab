'''
Linear regression using closed form solution without
regularization, including logarithmic transformation.

Error function: Mean-squared-error (MSE)

Written by Gustav Broe Hansen - s193855
Created on Sunday March 30 14:26 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

import sklearn.model_selection as ms

from PlotSettings import *
import DataExtraction as DE 

# Type of dataset
Type = 'Frame'  # 'Bar' or 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Different type of hydrib models
Types = {
    0 : 'Lin',
    1 : 'Log'
    }
# Selecting
RegType = Types[1]

# Normalizing the attributes dividing by the standard deviation
StdX = np.std(X, 0)
X = X * (1 / StdX)

# Printing dataset statistics to CLI
DE.DispStats(np.insert(X, X.shape[1], y, axis=1), AttName)


# %%
# --------------- FEATURE SELECTION ---------------

# Partitions (index array) for K-fold cross-validation
K = 14
CV = ms.KFold(n_splits=K, shuffle=True, random_state=721898)

# Logistic transformation of dataset
# (The translation has been iterativly estimated 
# in order to minimize numerical problems)
if RegType == 'Log':
    if Type == 'Bar':
        idx = AttName.index('Stiffness index')
        X[:, idx] = X[:, idx] + 0.2
        # (From NLR non-norm.=0.28 / norm.=0.013)
    elif Type == 'Frame':
        idx = AttName.index('No. interm.')
        X[:, idx] = X[:, idx] + 1E-6

# Adding offset
offs = np.exp(1) if RegType == 'Log' else 1
X = np.insert(X, 0, np.ones(N)*offs, axis=1)
AttName.insert(0, 'Offset')
AttSymb.insert(0, 'O')
N, M = X.shape

# Preallocating memory for coefficient and error measures
# Er = Error | Tr = Train | Te = Test
# FS = Featrure Selection | NF = No Features
Coef = np.zeros((K,M))
ErTr, ErTe = np.empty(K), np.empty(K)
ErTrFS, ErTeFS = np.empty(K), np.empty(K)
ErTrNF, ErTeNF = np.empty(K), np.empty(K)

# Linear regression
def LinReg(XTr, yTr):

    # Precomputing terms
    Xty = XTr.T @ yTr
    XtX = XTr.T @ XTr

    # Finding the optimal weights
    w = np.linalg.solve(XtX, Xty).squeeze()

    return w

j = 0
for TrInd, TeInd in CV.split(X):

    # Assigning current Training and Testing sets
    XTr, yTr = X[TrInd,:], y[TrInd]
    XTe, yTe = X[TeInd,:], y[TeInd]

    # Errors (sum-of-squares) when always guessing the mean of y
    # (meaning no features are used)
    ErTrNF[j] = ((yTr-yTr.mean())**2).sum()/len(yTr)
    ErTeNF[j] = ((yTe-yTe.mean())**2).sum()/len(yTe)
    
    # Linear regression
    if RegType == 'Lin':
        Coef[j, :] = LinReg(XTr, yTr)
        yEst = lambda x : np.sum(Coef[j, :]*x, axis=1)

    # Power regression
    if RegType == 'Log':
        Coef[j, :] = LinReg(np.log(XTr), np.log(yTr))
        yEst = lambda x : np.prod(x**Coef[j,:], axis=1)

    # Computing estimates and errors
    yTrEst, yTeEst = yEst(XTr), yEst(XTe)
    ErTr[j] = ((yTr - yTrEst)**2).sum()/len(yTr)
    ErTe[j] = ((yTe - yTeEst)**2).sum()/len(yTe)

    # Saving coefficients
    Coef[j, :]

    # Adding to the counter
    j += 1

# Adjusting x-ticks in bok plots
xt = AttName if Type == 'Bar' else [f'${a}$' for a in AttSymb]
    
# Visulization of the attributes using box plot
BoxPlot(X[:,1:], xt[1:], 'Attributes',
        'Observed values', y)

# Plot performance, comparing estimator with real values
# for the last fold
PerfPlot(yTe, yTeEst, AttUnit)
plt.savefig(f'Figures/{Type}/Performance/Perf{RegType}Reg.pdf', 
            format='pdf', bbox_inches='tight')

# Printing overview of results to CLI
PerfDisp(ErTrNF, ErTeNF, ErTr, ErTe, RegType)


# Printing coefficient statistics across folds
DE.DispStats(Coef, AttName[:-1])

# Data visualization of coe. using box plot
BoxPlot(Coef, xt[:-1], 'Attributes', 
        'Optimal weights from each fold, $w^{\\star}$')


# Plot performance, for baseline model (no features)
PerfPlot(yTe, np.full(len(yTe), yTe.mean()), AttUnit)

# Printing the figure
plt.savefig(f'Figures/{Type}/Performance/PerfBaseline.pdf', 
            format='pdf', bbox_inches='tight')

# Showing figures
plt.show()


