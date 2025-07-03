"""
Extracts data from the .cvs created by Colibri in 
Grasshopper which is outputof parametric analysis.

Written by Gustav Broe Hansen - s193855
Created on Sunday March 25 13:14 2025
"""

# ----------------- INITIALIZATION -----------------

import numpy as np
import pandas as pd     # Used to read csv-files



# ------------------- EXTRACT DATA -------------------

def ImportData(Type: str):

    # Set path to the datafile
    filename = 'Data/' + Type + 'Data.csv'

    # Loading data from csv file
    df = pd.read_csv(filename)

    # Converting th DataFrame in numpy arrays
    raw_data = df.values

    # Charecteristics of the data set based on Type
    # - Number of attributes based on Type
    # - Initial attribute info as a list of [unit, symbol] pairs
    if Type == 'Bar':
        M = 4
        AttInfo = [
            ['m', 'L'],     ['1', 'C'],
            ['1', '-'],     ['mm', 'h'],
            ['m', 'u']
        ]

    elif Type == 'Frame':
        M = 19
        AttInfo = [
            ['m', 'L'],         ['m', 'H'],
            ['GPa', 'E_{f}'],   ['GPa', 'E_{im}'],
            ['mm', 'b'],        ['mm', 'h_{c,B}'],
            ['mm', 'h_{c,C}'],  ['mm', 'h_{b,C}'],
            ['mm', 'h_{b,A}'],  ['1', 'C_{c,B}'],
            ['1', 'C_{C}'],     ['1', 'C_{b,A}'],
            ['1', 'C_{im,B}'],  ['GPa', 'E_{br}'],
            ['mm^2', 'A_{br}'], ['mm', 'h_{im}'],
            ['m', 's_{im}'],   [r'\degree', r'\alpha'],
            ['1', 'n_{im}'],    ['mm', 'u_{C}']
        ]

    # Split AttInfo into AttUnit and AttSymb
    AttUnit = ['\\mathrm{'+info[0]+'}' for info in AttInfo]
    AttSymb = [(info[1].replace('_{', '_{\\mathrm{')
                ).replace('}', '}}') for info in AttInfo]

    # Extracting the data matrix, X
    cols = range(M)
    X = raw_data[:, cols]
    y = raw_data[:, M].astype(float)

    # Number of additional attributes (minus dependent variable)
    naa = np.shape(df.columns)[0] - M - 1

    # Converting csv header to attribute names
    # (including the dependent variable y)
    AttName = [str(item).split(r':')[1] for item in df.columns[:-naa]]
    
    # Add '.' after 'interm' if it is not there
    # (Colibri replaces '.' with ' ' when creating the csv file)
    for i, name in enumerate(AttName):
        if 'interm ' in name or 'No ' in name:
            AttName[i] = (name.replace('interm ', 'interm.')
                          ).replace('No ', 'No.')

    # Number of data objects
    N = len(y)



    # -------------- PREPROCESSING OF DATA --------------

    # Bracing by Type
    if Type == 'Bar':

        # Adjusting unit [mm -> m]
        y = y*1E-3

        # List of possible strength class
        StrClass = ['C14', 'C16', 'C18', 'C20', 'C22', 
                    'C24', 'C27', 'C30', 'C35', 'C40', 
                    'C45', 'C50']

        # Strength class index
        SCInd = AttName.index('Strength class')

        # Assosiated mean modulus of elasticity parallel bending
        E0mean = np.array([7.0, 8.0, 9.0, 9.5, 10.0, 
                            11.0, 11.5, 12.0, 13.0, 
                            14.0, 15.0, 16.0])

        # Assosiated mean shear modulus
        Gmean = np.array([0.44, 0.50, 0.56, 0.59, 
                            0.63, 0.69, 0.72, 0.75, 
                            0.81, 0.88, 0.94, 1.00])

        # Combining lists in directory
        CE = dict(zip(StrClass, E0mean))

        # Vector with modulus of elasticity instead of CS
        NewCol = [CE[C] if C in CE else None for C in X[:,SCInd]]

        # Inserting the new vector
        X = np.insert(X, SCInd+1, NewCol, axis=1)

        # Adding information about the new attribute
        NewAtt = ['Elasticity modulus', 
                  r'E_{\mathrm{0,mean}}', r'\mathrm{GPa}']
        [List.insert(SCInd+1, Val) for List, Val 
         in zip([AttName, AttSymb, AttUnit], NewAtt)]


        # Repeating for mean shear modulus 
        # (added to perform feature selection)
        # CE = dict(zip(StrClass, Gmean))
        # NewCol = [CE[C] if C in CE else None for C in X[:,SCInd]]
        # X = np.insert(X, SCInd+1, NewCol, axis=1)
        # NewAtt = ['Shear modulus', r'G_{\mathrm{mean}}', r'\mathrm{GPa}']
        # [List.insert(SCInd+1, Val) for List, Val in zip([AttName, AttSymb, AttUnit], NewAtt)]


        # Removing the strength class attribute
        X = np.delete(X, SCInd, axis=1)
        [List.pop(SCInd) for List in [AttName, AttSymb, AttUnit]]

        # Converting the X-matrix to type float
        # (Initially it was object-type since it contained str)
        X = X.astype(float)

        # Deflection limit of each object
        DeflLim = X[:, AttName.index('Length')]/100

        # Keeping only the reasonable entires
        y, X = y[y < DeflLim], X[y <  DeflLim]

        # Converting deflection to transverse flexibility 
        # (applied UDL is defined in GH as: q = 1 [kN/m])
        y = y*1E3/1
        AttName[-1] =  'Transverse flexibility'
        AttSymb[-1] = r'\frac{u}{q}'
        AttUnit[-1] = r'\frac{\mathrm{mm}}{\mathrm{kN/m}}'


    elif Type == 'Frame':

        # ------------- Feature transformation ------------- 

        # Converting the X-matrix to type float
        X = X.astype(float)

        # Find index of 'b' and all indices of 'h' attributes
        b_idx = AttSymb.index('b')
        h_idx = [i for i, s in enumerate(AttSymb) if 'h_' in s]

        # Compute moment of inertia and insert after 'h'
        for i in h_idx:

            # Computing second moment of area, strong axis
            # (unit conversion: mm^4 -> 1E8*mm^4)
            I_y = 1/12*X[:, b_idx]*X[:, i]**3 *1E-8

            # Index to insert after the 'h' attribute
            # (accounting for previous insertions)
            # idx = i + 1 + offset  
            X = np.insert(X, i+1, I_y, axis=1)
            AttName.insert(i+1, AttName[i].replace('Height', 'Inertia'))
            AttSymb.insert(i+1, AttSymb[i].replace('h', 'I', 1))
            AttUnit.insert(i+1, r'10^{8}\,\mathrm{mm^4}')

            # Remove the old 'h' attributes 
            X = np.delete(X, i, axis=1)
            [List.pop(i) for List in [AttName, AttSymb, AttUnit]]

        # Remove the width attribute 
        X = np.delete(X, b_idx, axis=1)
        [List.pop(b_idx) for List in [AttName, AttSymb, AttUnit]]

        # No bracing when no intermediate beams
        # Set 'Material bracing' and 'Area bracing' to zero when 'Spacing interm.' == 0
        # mask = X[:, AttName.index('No. interm.')] == 0
        # X[mask, AttName.index('Material bracing')] = 0
        # X[mask, AttName.index('Area bracing')] = 0

        # Extract relevant attributes
        E = X[:, AttName.index('Material bracing')]
        A = X[:, AttName.index('Area bracing')]
        H = X[:, AttName.index('Height')]
        s_im = X[:, AttName.index('Spacing interm.')]
        alpha = X[:, AttName.index('Pitch')]

        # Calculate bracing height, length, and axial stiffness
        h_br = np.where(E == 210, H + s_im * np.tan(alpha * np.pi/180), H)
        L_br = np.sqrt(s_im**2 +  h_br**2)
        k_br = E * A / L_br  # [GPa]*[mm^2]/[m] -> [kN/m]
        k_br = k_br *1E-3  # [kN/m] -> [MN/m]

        # Computing the angle to horizontal
        a_br = np.arctan(h_br / s_im) * 180/np.pi

        # Replace 'Area bracing' and 'Material bracing' with new attributes
        replacements = [
            ('Area bracing', 'Axial stiff. bracing', 
             r'k_\mathrm{br}', r'\mathrm{\frac{MN}{m}}', k_br),
            ('Material bracing', 'Angle bracing', 
             r'\alpha_\mathrm{br}', r'\mathrm{\degree}', a_br)
        ]
        for name, n_name, n_symb, n_unit, n_val in replacements:
            idx = AttName.index(name)
            X[:, idx] = n_val
            AttName[idx] = n_name
            AttSymb[idx] = n_symb
            AttUnit[idx] = n_unit


        # # Removing "junk" attributes after feature selection
        junk = ['Material interm.', 'SI beam-apex', 
            'SI interm.-base', 'Inertia interm.']
        for attr in junk:
            idx = AttName.index(attr)
            X = np.delete(X, idx, axis=1)
            [List.pop(idx) for List in [AttName, AttSymb, AttUnit]]


        # Deflection limit of all object
        DeflLim = 50           # [mm]

        # Keeping only the reasonable entires
        y, X = y[y < DeflLim], X[y < DeflLim]


        # Applied peak wind pressure and reach from GH
        p_p = 0.9*6             # [kN/m^2]*[m]

        # Converting deflection to horizontal flexibility
        y = y/p_p               # [mm]/[kN/m]

        # Redefining its attributes
        AttName[-1] =  'Horizontal flexibility'
        AttSymb[-1] = r'u/q_p\,s' # Alternative: r'\frac{u}{q_p\,s}'
        AttUnit[-1] = r'\frac{\mathrm{mm}}{\mathrm{kN/m}}'



    # # Exports the feature matrix X and target vector y to a CSV file
    # filename='ExportedData.csv'
    # data = np.column_stack((X, y))
    # df = pd.DataFrame(data, columns=AttName)
    # df.to_csv(filename, index=True)


    # Updating the number of objects and attributes
    N, M = np.shape(X)

    # Standardizing the dataset by first subtracting each
    # attribute with their respective mean
    mu = X.mean(axis=0)
    X_tilde = X - np.ones((N, 1)) * mu

    # then dividing by the standard deviation
    sigma = np.std(X_tilde, axis=0)
    X_tilde = X_tilde * (1 / sigma)
    # (alternativly use spicy.zscore(X))

    return X, y, AttName, AttSymb, AttUnit, N, M, X_tilde


def DispStats(X, AttName):
    '''Function to print the statistical properties of 
    each column in a dataset (np.array) in CLI.'''

    # Printing information about the dataset
    Stats = zip(np.min(X, axis=0), np.max(X, axis=0),
            np.mean(X, axis=0), np.std(X, axis=0))

    # Define column width
    CW = [24, 10, 10, 10, 10]

    # Print table header
    header = ["\nAttribute", "Minimum", "Maximum", "Mean", "Std. dev."]
    print("".join(f"{h:<{CW[i]}}" for i, h in enumerate(header)))
    print("-" * sum(CW))

    # Print each row with fixed width
    for Att, Val in zip(AttName, Stats):
        print(f"{Att:<{CW[0]}}" + "".join(f"{v:<{CW[i+1]}.3g}" for i, v in enumerate(Val)))
