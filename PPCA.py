def PPCA(theta, n ,nbins):
    # Function implementing PPCA algorithm
    
    # Setting the weights
    w_l, w_u, w_bn = setting_weights(theta, n ,nbins)

    # Creating Matrices
    W, C_M, R_M, B_B, B = mean_matrices()
    R_V, b_i = basis_matrices()

    # Initialization
    mu, p_bv = initialization()

    # Solver
    mu, p1, p2, betas, history = solver()

