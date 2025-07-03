import torch
import numpy as np

def TrainANN(XTr, yTr, NoHU=7, MaxIte=49000, RCTol=1E-6):
    r''' 
    Trains an artificial neural network (ANN) using the PyTorch-module,
    and returns the 

    Inspired by 'train_neural_net' from the 'dtuimldmtools'-modulue
    Located in "...\Lib\site-packages\dtuimldmtools\models\nn_trainer.py"
    Created by DTU Compute (https://pypi.org/project/dtuimldmtools/)
    '''

    # Number of features in the training set
    N, M = XTr.shape

    # Defining loss function, mean-squared-error
    LossFn = torch.nn.MSELoss()

    # Defining the characteristics and structure of the ANN
    model = lambda: torch.nn.Sequential(

        # Connecting M attributes to a number of hidden units
        torch.nn.Linear(M, NoHU),

        # Assigning transfer function to hidden layer, tangent hyperbolic
        torch.nn.Tanh(),

        # Trying something
        torch.nn.Linear(NoHU, int(NoHU/2)),
        torch.nn.Tanh(),

        # Connecting hidden units to a single output
        torch.nn.Linear(int(NoHU/2), 1),

        # Does not assign the out layer a transfer function, linear
    )

    # Initializing ANN and weights of each layer, since the hidden layer 
    # uses a 'tanh()' transfer function it is recommended to use an 
    # uniform Xavier distributes, that scales with the number of 
    # in- and outputs, which increases the change of convergence.
    # [https://pytorch.org/docs/2.6/nn.init.html]
    # (Weights are of type torch.float32, not 64 for efficiency)
    ANN = model()
    Gain = torch.nn.init.calculate_gain('tanh')
    [torch.nn.init.xavier_uniform_(ANN[j].weight, gain=Gain) for j in [0, 2]]

    # Unpacking all parameters of the ANN
    Par = [p for (name, p) in ANN.named_parameters()]

    # Assinging on optimization algorithim to parameters
    # Most commonly optimal weights are found using stochastic gradient
    # descent (SGD) which step-wise update the weights using a fixed 
    # learning rate, the optimal amount is specific to each problem.
    # An extension of SGD called the Adam-al g. is used instead, where 
    # the learning rate is adjusted automatically, without regularization.
    Optim = torch.optim.Adam([{'params' : Par}], weight_decay=0)

    # Allocating space for the loss history
    LF = 100
    LossHist = np.zeros([int(MaxIte/LF), 2])

    # Setting the initial error to a large number
    LossPrev = 1E7

    # Printing header for progress
    print(f"|{'Ite.':<7}|{'Loss':<10}|{'RelImp.':<10}|")

    # Training the NN given the above definition
    for j in range(MaxIte):

        # Estimating the dependent variable
        yEst = ANN(XTr)

        # Computing the loss and saving to history
        Loss = LossFn(yEst, yTr)
        LossCur = Loss.item()

        # Printing the progress
        if j % LF == 0:
            LossHist[int(j/LF), :] = [j, LossCur]

        # Relative improvement in error/loss and
        RelLoss = np.abs(LossCur - LossPrev)/LossPrev
        LossPrev = LossCur

        # Checking if the solution has converged
        if RelLoss <= RCTol:
            break

        # Printing the progress
        if j % 5000 == 0:
            print(f'|{j:<7.0f}|{LossCur:<10.4g}|{RelLoss:<10.2e}|')
            
        # Gradients are reset and updated by backwards propagation 
        # of the error through the units and computing the derivatives.
        Optim.zero_grad()
        Loss.backward()

        # Performs optimization step
        Optim.step()

    # Printing the resulting error
    print('-'*(7+10+10+4))
    print(f'|{j:<7.0f}|{LossCur:<10.4g}|{RelLoss:<10.2e}|')
    
    # Removing remaining empty elements
    LossHist = np.delete(LossHist, LossHist[:,0]==0, axis=0)

    return ANN, LossHist