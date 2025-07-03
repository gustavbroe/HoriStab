'''
Non-linear regression using least squared optimization.
It would also be possible use gradient decsent to optimize
the weights.

Error function: Sum-of-squares

Written by Gustav Broe Hansen - s193855
Created on Sunday March 30 14:26 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

import sklearn.linear_model as lm
import sklearn.model_selection as ms
from scipy.optimize import least_squares

from PlotSettings import *
import DataExtraction as DE

# Type of dataset
Type = 'Frame'  # 'Bar' or 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Overview of indeis and names of attributes
[print(f'{i}|{name}') for i, name in enumerate(AttName)]



# %%
# ------------- FEATURE TRANSFORMATION AND DATA SUBSETS -------------

# Transforming the pitch angle to cosine
# idx = AttName.index('Pitch')
# X[:, idx] = np.sin(np.deg2rad(X[:, idx]))

# # Keeping only the attributes with no intermediate columns
# mask = X[:, AttName.index('No. interm.')] == 0
# X = X[mask]
# y = y[mask]

# Remove all objects with any 'SI' attribute less than 0.1
# si_indices = [i for i, name in enumerate(AttName) if 'SI' in name]
# if si_indices:
#     mask_si = np.all(X[:, si_indices] >= 0.7, axis=1)
#     X = X[mask_si]
#     y = y[mask_si]

# Printing resulting  shape of the dataset
print(np.shape(X))


# %%
# --------------- NONLINEAR REGRESSION ---------------

# Partitions (index array) for K-fold cross-validation
K = 10
CV = ms.KFold(n_splits=K, shuffle=True, random_state=217)

# Normalizing the attributes dividing by the standard deviation
StdX = np.std(X, 0)
X = X * (1 / StdX)

# Additng an offset to the attributes
# (The translation has been iterativly estimated 
# in order to minimize numerical problems)
if Type == 'Bar':
    idx = AttName.index('Stiffness index')
    X[:, idx] = X[:, idx] + 2E-1
# elif Type == 'Frame':
#     idxs = [i for i, name in enumerate(AttName) if 'SI' in name]
#     for idx in idxs:
#         X[:, idx] = X[:, idx] + 2E-1
#         # X[:, idx] = np.log(X[:, idx])

# Printing dataset statistics to CLI
DE.DispStats(X, AttName[:-1])

# Adding offset
def AddOffs(X, scalar):

    # Inserting a column of ones infront, assinging name and updating
    X = np.insert(X, 0, np.ones(X.shape[0])*scalar, axis=1)
    AttName.insert(0, 'Offset')
    return X


if Type == 'Bar':
    # Different types of hybrid models
    Types = {
        0 : 'SumPower',
        1 : 'SumPowerOffset',
        2 : 'ProdPower',
        3 : 'ProdPowerOffset',
        4 : 'FromTheory',
        5 : 'LinFeatTrans',
        6 : 'Best',
        }
    # Selecting
    RegType = Types[6]

    # Suggested expression for computing y with corresponding
    # initial guess of the weights from data visualizations
    if RegType == 'SumPower':
        w0 = np.array([1.8, -1.2, -0.4, -0.6])
        yEst = lambda w, X : (
            X[:,0]**w[0] + X[:,1]**w[1] + 
            X[:,2]**w[2] + X[:,3]**w[3]
            )    
    elif RegType == 'SumPowerOffset':
        X = AddOffs(X, 1)
        w0 = np.array([7.0, 1.8, -1.2, -0.4, -0.6])
        yEst = lambda w, X : (
            X[:,0]*w[0]  + X[:,1]**w[1] + X[:,2]**w[2] + 
            X[:,3]**w[3] + X[:,4]**w[4]
            )    
    elif RegType == 'ProdPower':
        w0 = np.array([3.8, -0.2, -1.0, -2.8])
        yEst = lambda w, X : (
            X[:,0]**w[0] * X[:,1]**w[1] * 
            X[:,2]**w[2] * X[:,3]**w[3]
            )    
    elif RegType == 'ProdPowerOffset':
        X = AddOffs(X, np.exp(1))
        w0 = np.array([7.0, 3.8, -0.2, -1.0, -2.8, 0.2])
        yEst = lambda w, X : (
            X[:,0]**w[0] * X[:,1]**w[1] * (X[:,2]+w[5])**w[2] * 
            X[:,3]**w[3] * X[:,4]**w[4]
            )    
    elif RegType == 'FromTheory':
        X = AddOffs(X, 1.0)
        w0 = np.array([1.0, -1.0, 0.8, 0.4, 0.7, 0.2, 1.4])
        yEst = lambda w, X : (
            w[0]*X[:,0] + w[1]*X[:,1]**2/(X[:,4]**3*X[:,3]*X[:,2])*(
                w[2]*X[:,2]*X[:,1]**2 + w[3]*X[:,1]**2 +
                w[4]*X[:,3]*X[:,1]**2 + w[5]*X[:,3]**2*X[:,1]**2
            ) + w[6]*X[:,1]**4/(X[:,4]**3*X[:,3])
            )    
    elif RegType == 'LinFeatTrans':
        X = AddOffs(X, 1.0)
        w0 = np.array([1.0, -1.0, -2.8, 4.0, 1.0, 4.0, 0.6, 0.5, 0.4])
        yEst = lambda w, X : (
            w[0]*X[:,0] + w[1]*X[:,1] + w[2]*X[:,2] + w[3]*X[:,3] +
            w[4]*X[:,4] + w[5]*X[:,1]**3 + w[6]*X[:,2]/X[:,1] + 
            w[7]*X[:,4]**3 + w[8]*X[:,3]*X[:,4]**3
        )    
    elif RegType == 'Best':
        X = AddOffs(X, np.exp(1))
        w0 = np.array([3, 3, -0.2, -0.8, -0.2, 0.3, -0.3, 0.1, -0.4, 0.1, 0.1])
        yEst = lambda w, X : (
            X[:,0]**w[0] * X[:,1]**w[1] * X[:,2]**w[2] * 
            X[:,3]**w[3] * X[:,4]**w[4] + 
            X[:,2]**3*w[5] + X[:,2]**4/X[:,1]*w[6] + X[:,2]**4/X[:,4]*w[7] +
            X[:,1]**2*w[8] + X[:,4]**3*w[9] + X[:,3]/X[:,2]**3*w[10]
        )

elif Type == 'Frame':
    
    # Different types of hybrid models
    Types = {
        0 : 'CC',
        1 : 'CC2',
        2 : 'CC3',
        }
    # Selecting
    RegType = Types[2]

    # NOTE: CC and CC2 requires subset where 'No. interm.' == 0

    # Suggested expression for computing y with corresponding
    # initial guess of the weights from data visualizations
    if RegType == 'CC':
        # X = AddOffs(X, 1.0)
        # w0 = np.random.uniform(-3, 3, 12)
        # w0 = np.ones(11)
        #     0     1     2     3     4     5     6     7
        w0 = [ 0.1,  3.0,  0.4,  0.4, -1.0,  1.0,  0.3, -0.5]
        # Positive = Higher is more
        # Negative = Higher is less
        yEst = lambda w, X : (
            w[0] * X[:,1]**w[1] * (1/(X[:,7]**w[2] + 1) + 1/(X[:,8]**w[3] + 1))
            * X[:,3]**w[4] * ((X[:,0]*X[:,12])**w[5] + 1) * X[:,0]**w[6] * X[:,2]**w[7]
            )
    elif RegType == 'CC2':
        #     0     1     2     3     4     5     6     7     8     9     10
        w0 = [ 0.1,  3.0, -0.4,  0.4,  0.4, -1.0, -0.3, -0.5, -1.0, -0.5,  0.4]
        # Positive = Higher is more
        # Negative = Higher is less
        yEst = lambda w, X : (
            w[0] * X[:,1]**w[1] * X[:,2]**w[2] * (
                  1/(X[:,7]**w[3] + 1) * X[:,3]**w[5]
                + 1/(X[:,8]**w[4] + 1) * X[:,5]**w[6] * X[:,0]**w[9] * (X[:,12]**w[7] + 1)
             ) * X[:,4]**w[8]
            )
    elif RegType == 'CC3':
        #        0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
        w0 = [-4.0, 0.5, 2.0,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2, 0.2,-0.2,-0.2, 0.2,-0.2,0.01,0.01]
        # Positive = Higher is more
        # Negative = Higher is less
        yEst = lambda w, X : (
            np.exp(1)**w[0] * X[:,0]**w[1] * X[:,1]**w[2] *  X[:,2]**w[3] * 
            X[:,3]**w[4] * X[:,4]**w[5] *  X[:,5]**w[6] * X[:,6]**w[7] *
            (X[:,7] + 1E-3)**w[8] * (X[:,8] + 1E-3)**w[9] *  X[:,11]**w[10] * X[:,12]**w[11] 
             + w[12] * X[:,9]**w[13] * X[:,10]**w[14] * (1 - 0**X[:,13])
            )

# Expression for the error between actual and estimated y
Err = lambda w, X, y : y - yEst(w, X)

# Updating sizes
N, M = X.shape

# Preallocating memory for coefficient and error measures
# Er = Error | Tr = Train | Te = Test
# FS = Featrure Selection | NF = No Features
Coef = np.zeros((K,len(w0)))
ErTr, ErTe = np.empty(K), np.empty(K)
ErTrFS, ErTeFS = np.empty(K), np.empty(K)
ErTrNF, ErTeNF = np.empty(K), np.empty(K)

j = 0
for TrInd, TeInd in CV.split(X):

    # Assigning current Training and Testing sets
    XTr, yTr = X[TrInd,:], y[TrInd]
    XTe, yTe = X[TeInd,:], y[TeInd]

    # Errors (sum-of-squares) when always guessing the mean of y
    # (meaning no features are used)
    ErTrNF[j] = ((yTr-yTr.mean())**2).sum()/len(yTr)
    ErTeNF[j] = ((yTe-yTe.mean())**2).sum()/len(yTe)

    # Non-linear least squares to find the optimal weights
    res = least_squares(Err, w0, args=(XTr, yTr), method='lm')

    # Saving coefficients
    w = res.x
    Coef[j,:] = w

    # Reverting transformation
    yTrEst, yTeEst = yEst(w,XTr), yEst(w,XTe)
    ErTr[j] = ((yTr - yTrEst)**2).sum()/len(yTr)
    ErTe[j] = ((yTe - yTeEst)**2).sum()/len(yTe)

    # Adding to the counter
    j += 1


# Printing coefficient statistics across folds
if Type == 'Bar':
    if RegType in [Types[4], Types[5], Types[6]]:
        [print(f'{coe:0.3f}', end=', ') for coe in np.mean(Coef, axis=0)]
    else:
        DE.DispStats(Coef, AttName[:-1])

print('\n')
[print(f'{i:0.0f}|{coe:0.2f}') for i, coe in enumerate(np.mean(Coef, axis=0))]
[print(f'{coe:0.2f}',end=', ') for coe in np.mean(Coef, axis=0)]
print('\n')

# Plot performance, comparing estimator with real values
# for the last fold
PerfPlot(yTe, yTeEst, AttUnit, color=Col['B'])

# Printing the figure
plt.savefig(f'Figures/{Type}/Performance/Perf{RegType}Reg.pdf', 
            format='pdf', bbox_inches='tight')

# Printing overview of results
PerfDisp(ErTrNF, ErTeNF, ErTr, ErTe, RegType)

# Showing figures
plt.show()



# ------ ADDITIONAL SCRIPTS TO AID TROUBLE-SHOOTING EXPRESSIONS ------ 

# # Parity plot with object indices used to identify outliers
# plt.figure()
# plt.scatter(yTe, yTeEst, c='b', alpha=0.7)
# for i, (yt, yest) in enumerate(zip(yTe, yTeEst)):
#     plt.text(yt, yest, str(i), fontsize=8, ha='right', va='bottom')
# plt.xlabel('True y (yTe)')
# plt.ylabel('Estimated y (yTeEst)')
# plt.title('Scatter plot of yTe vs yTeEst with indices')
# plt.grid(True)
# plt.tight_layout()

# # Identify the index and value of the highest estimated y in the test set
# max_idx = np.argmin(yTeEst)
# print(f'Highest yTeEst: {yTeEst[max_idx]:.4f} at index {max_idx}')
# print('Corresponding X object:')
# for i, (name, value) in enumerate(zip(AttName, XTe[max_idx])):
#     print(f'{i}. {name}: {value:.2g}')