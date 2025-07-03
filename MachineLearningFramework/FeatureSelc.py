'''
Forward sequential feature selection based on standard 
linear regression using K-fold cross-valibration, which 
aims to identify possible junk-attributes. 

Error function: Mean-squared-error (MSE)

Written by Gustav Broe Hansen - s193855
Created on Sunday March 30 14:26 2025
'''

# %% 
# ----------------- INITIALIZATION -----------------

import sklearn.linear_model as lm
import sklearn.model_selection as ms

from dtuimldmtools import feature_selector_lr

from PlotSettings import *
import DataExtraction as DE

# Type of dataset: 'Bar' or 'Frame'
# (Ensure "Junk"-attributes are not removed in DataExtraction)
Type = 'Frame'
X, y, AttName, AttSymb, AttUnit, N, M, X_tilde = DE.ImportData(Type)

# Normalized dataset
X = X_tilde



# --------------- FEATURE SELECTION ---------------

# Type specific settings
# - Relative height of the figure
# - Number of K-fold cross-validation
# - Configuration of subplot in single outerfold
if Type == 'Bar':
    K = 14
    rh = 0.3
elif Type == 'Frame':
    K = 22
    rh = 0.5

# Partitions (index array) for K-fold cross-validation
CV = ms.KFold(n_splits=K, shuffle=True, random_state=217)

# Preallocating memory for features and error measures
# Er = Error | Tr = Train | Te = Test
# FS = Featrure Selection | NF = No Features
Feat = np.zeros((M,K))
ErTr, ErTe = np.empty(K), np.empty(K)
ErTrFS, ErTeFS = np.empty(K), np.empty(K)
ErTrNF, ErTeNF = np.empty(K), np.empty(K)

j = 0
for TrInd, TeInd in CV.split(X):

    # Assigning current Training and Testing sets
    XTr, yTr = X[TrInd,:], y[TrInd]
    XTe, yTe = X[TeInd,:], y[TeInd]

    # Number of internal cross-validation folds
    K_int = 7

    # Errors (sum-of-squares) when always guessing the mean of y
    # (meaning no features are used)
    ErTrNF[j] = ((yTr-yTr.mean())**2).sum()/len(yTr)
    ErTeNF[j] = ((yTe-yTe.mean())**2).sum()/len(yTe)
    
    # Errors when the dataset is unchanged
    # (meaning all features are included)
    Mod = lm.LinearRegression(fit_intercept=True).fit(XTr, yTr)
    ErTr[j] = ((yTr-Mod.predict(XTr))**2).sum()/len(yTr)
    ErTe[j] = ((yTe-Mod.predict(XTe))**2).sum()/len(yTe)

    # Calling function to perform feature selection
    # "...\anaconda3\envs\MP\Lib\site-packages\dtuimldmtools
    #  \crossvalidation\implementations.py"
    FeatSel, FeatRec, LossRec = feature_selector_lr(
        XTr, yTr, K_int, loss_record=[ErTrNF[j]])

    # Training and testing linear model with selected features
    Mod = lm.LinearRegression(fit_intercept=True).fit(XTr[:,FeatSel], yTr)
    ErTrFS[j] = ((yTr-Mod.predict(XTr[:,FeatSel]))**2).sum()/len(yTr)
    ErTeFS[j] = ((yTe-Mod.predict(XTe[:,FeatSel]))**2).sum()/len(yTe)

    # Converting to bitmap (for plotting)
    Feat[FeatSel,j] = 1

    # Adding to the counter
    j += 1


# Initializing figure for final results
cm = 1/2.54
fig, ax = plt.subplots(1, 1, figsize=(21.0*cm*0.8, 
        29.7*cm*0.8*rh), layout='constrained')

# Naming and positioning of figure window
fig.canvas.manager.set_window_title("Feature selection")
# fig.canvas.manager.window.wm_geometry("+100+100")

# Plotting the selected features from each K-fold
FSplot(Feat, AttName[:-1], range(1,K+1), ax)
plt.xlabel('Outer fold')
plt.ylabel('Name of attribute')

# Printing the figure
plt.savefig(f'Figures/{Type}/{Type}FeatureSelc.pdf', format='pdf', 
            bbox_inches='tight')


# %%
# Initializing figure for final outer fold results with left axis half the total height
fig, ax = plt.subplots(1, 2, figsize=(21.0*cm*0.8, 29.7*cm*0.8
                        *(0.3 + (rh - 0.3)*0.4)), layout='tight')

# Naming and positioning of figure window
fig.canvas.manager.set_window_title("Feature selection")
# fig.canvas.manager.window.wm_geometry("+100+600")

# Plotting the validation/testing error for iterations
xTicks = range(0, FeatRec.shape[1])
ax[0].plot(range(0, len(LossRec)), LossRec, c=Col['G'], lw=1.4)
ax[0].grid(True)
ax[0].set_xticks(list(xTicks))
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Mean squared testing error \n'
                 fr'$\left(K_2 = {K_int:.0f}\right)$')

# Plotting the feature set for each iteration
yTicks = [AttName[:-1] if Type == 'Bar' else 
          [fr'${symb}$' for symb in AttSymb[:-1]]][0]
FSplot(FeatRec[:, 0:], yTicks, xTicks, ax[1])
ax[1].set_xlabel('Iteration')

# Adjusting for readability
if Type == 'Frame':
    ax[0].set_xticks(list(xTicks)[::2])
    ax[1].set_xticks(list(xTicks)[::2])
    ax[1].tick_params(axis='y', labelsize=10)

# Printing the figure
plt.savefig(f'Figures/{Type}/{Type}FeatureSelcSingle.pdf', format='pdf', 
            bbox_inches='tight')


# %%
# Printing overview of results
# Define a helper function to print errors and R^2
def print_res(title, train_err, test_err, train_base, test_base):
    print(title)
    print(f'- Training error: {train_err.mean():.2f}')
    print(f'- Test error:     {test_err.mean():.2f}')
    print(f'- R^2 train:     {1 - train_err.sum()/train_base.sum():.3f}')
    print(f'- R^2 test:     {1 - test_err.sum()/test_base.sum():.3f}')

print_res('Only guessing the mean value of y:', 
              ErTrNF, ErTeNF, ErTrNF, ErTeNF)
print_res('Linear regression without feature selection:', 
              ErTr, ErTe, ErTrNF, ErTeNF)
print_res('\nLinear regression with feature selection:', 
              ErTrFS, ErTeFS, ErTrNF, ErTeNF)

print('\nCoefficients of the last fold:')
for Att, Coe in zip(np.array(AttName)[FeatSel], Mod.coef_):
    print(f" - {Att} = {Coe:.2f}")

# Showing figures
plt.show()