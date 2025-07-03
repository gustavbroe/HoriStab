# Euler–Bernoulli beam theory
def EB(x, q, L, E, I_y, K_A, K_B):
    
    # Stiffness indices
    C_A = K_A*L/(E*I_y)
    C_B = K_B*L/(E*I_y)

    # Intermediate parameter
    eta = (C_B + 6)/(C_A*C_B + 4*C_A + 4*C_B + 12)

    # Computing transverse deflection
    w = (q*L**4)/(24*E*I_y) * ( 
        2*eta*x/L 
        + C_A*eta * (x/L)**2 
        - (1 + 2*eta + C_A*eta) * (x/L)**3 
        + (x/L)**4 
    )

    return w


# Timoshenko–Ehrenfest beam theory
def TE(x, q, L, E, I_y, K_A, K_B, A_s, G):
    
    # Stiffness indices
    C_A = K_A*L/(E*I_y)
    C_B = K_B*L/(E*I_y)
    C_s = E*I_y/(A_s*G*L**2)

    # Intermediate parameter
    eta_s = (72*C_A*C_B*C_s**2 + 6*C_A*C_B*C_s 
            + 72*C_A*C_s**2 + 72*C_B*C_s**2 
            + 30*C_A*C_s + 30*C_B*C_s + C_B 
            + 72*C_s + 6)/(12*C_A*C_B*C_s 
            + C_A*C_B + 12*C_A*C_s + 12*C_B
            *C_s + 4*C_A + 4*C_B + 12)

    # Computing transverse deflection
    w = (q*L**4)/(24*E*I_y) *( 
        2*eta_s*x/L 
        - (((6*eta_s + 3)*C_s - eta_s)*C_A + 12*C_s)/(3*C_A*C_s + 1) * (x/L)**2
        - ((C_A + 2)*eta_s - 12*C_s + 1)/(3*C_A*C_s + 1) * (x/L)**3
        + (x/L)**4
    )

    return w


# -------------- WELL-KNOWN SOLUTIONS --------------

# Pinned - Pinned
def PP(x, q, L, E, I_y):

    # Well-known solution
    w = (q*L**3*x/(24*E*I_y)*
        (1 - 2*(x/L)**2 + (x/L)**3))

    return w

# Fixed - Fixed
def FF(x, q, L, E, I_y):

    # Well-known solution
    w = (1/24*q*L**4/(E*I_y)*
        ((x/L)**2 - 2*(x/L)**3 
        + (x/L)**4))

    return w



# ----------- EXPRESSION FROM LITERATURE -----------

# Braun16
def Braun16(x, q, L, E, I_y, K_A, K_B):
    
    # 
    n = K_A*L/(E*I_y)
    m = K_B*L/(E*I_y)
    k_A = (m + 6)/(m + 4 + 4*m/n + 12/n)
    k_B = (n + 6)/(n + 4 + 4*n/m + 12/m)

    # 
    w = -q*(L - x)*((k_A + k_B/2 - 
        3/2)*L**2 - x*(k_A - k_B + 3)
        *L/2 + (3*x**2)/2)*x/(36*E
        *I_y)

    return w

# Villegas23
def Villegas23(x, q, L, E, I_y, K_A, K_B, A_s, G):
    
    # 
    r_i = K_A/(K_A + E*I_y/L)
    r_j = K_B/(K_B + E*I_y/L)
    phi = 12*E*I_y/(A_s*G*L**2)
    R = 12 - (8 - phi)*(r_i + r_j) + (5 - phi)*r_i*r_j

    # 
    w = q*((6 + (((-1 + r_j)*r_i - r_j)*phi**2)/2 
        + (-6 + ((9 - 7*r_j)*r_i)/2 + R + (9*r_j)/2)
        *phi + (5*r_j - 6)*r_i - 5*r_j)*L**2 - ((((
        -1 + r_j)*r_i - r_j)*phi - 5*(r_j - 6/5)*(-2 
        + r_i))*x*L)/2 - 2*(-9 + 3*((-1 + r_j)*r_i 
        - r_j)*phi/4 + 3*(2 - (5*r_j)/4)*r_i + R + 6
        *r_j)*x**2)*(L - x)*x/(12*E*I_y*R)

    return w


# --------------------- TE ENVELOPE ---------------------

# TE Pinned-Pinned
def TE_PP(x, q, L, E, I_y, A_s, G):
    
    # Shear index
    C_s = E*I_y/(A_s*G*L**2)

    # Computing transverse deflection
    w = (q*L**4)/(24*E*I_y) *( 
        (x/L)**4 - 2*(x/L)**3 - 12*(x/L)**2*C_s + (12*C_s + 1)*(x/L)
    )

    return w

# TE Fixed-Fixed
def TE_FF(x, q, L, E, I_y, A_s, G):
    
    # Shear index
    C_s = E*I_y/(A_s*G*L**2)

    # Computing transverse deflection
    w = (q*L**4)/(24*E*I_y) *( 
        (x/L)**4 - 2*(x/L)**3 - (x/L)**2*(12*C_s - 1) + 12*C_s*(x/L)
    )

    return w