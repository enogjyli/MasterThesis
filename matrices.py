import numpy as np
import matplotlib.pyplot as plt
import compute

def mean(n,nbins,D,m,w_bn,beta):
    # Function that computes the matrices necessary for mean update
    
    # Observation-specific matrix W_{i} dim = D x (nbins+1)*D
    W = []
    for i in range(n):
        W_temp = w_bn[0][i]*np.eye(D)    
        for j in range(1,nbins+1):
            W_temp = np.concatenate((W_temp, w_bn[j][i]*np.eye(D)),axis=-1)
        W.append(W_temp)

    #print(W_temp.shape)
    #print(W[5])

    # Weight-product matrix C_{(M),i} dim = D*(nbins+1) x D*(nbins+1)
    C_M = []
    for i in range(n):
        CM_temp = (w_bn[0][i]**2)*np.eye(D)
        # create first row block
        for s in range(1,nbins+1):           
            CM_temp = np.concatenate((CM_temp, (w_bn[0][i]*w_bn[s][i])*np.eye(D)),axis=-1)
        #print(CM_temp.shape)
        # create other row blocks
        for j in range(1,nbins+1):
            CM_temp_row = (w_bn[j][i]*w_bn[0][i])*np.eye(D)
            for s in range(1,nbins+1):
                CM_temp_row = np.concatenate((CM_temp_row, (w_bn[j][i]*w_bn[s][i])*np.eye(D)),axis=-1)
            # stack rows
            CM_temp = np.concatenate((CM_temp, CM_temp_row),axis=0)

        C_M.append(CM_temp)

    #print(CM_temp.shape)
    #print(C_M[0][2,:])

    # Matrix R_{(M)} dim = D*(nbins+1) x D*(nbins+1)
    R_M = 2*np.eye(D*(nbins+1))

    for i in range(D):    # first and last block
        R_M[i][i] = 1
        R_M[D*(nbins+1)-1-i][D*(nbins+1)-1-i] = 1 

    for i in range(1,nbins+1):
        for j in range(D):
            R_M[(i-1)*D+j][i*D+j] = -1
            R_M[i*D+j][(i-1)*D+j] = -1

    #print(R_M.shape)
    #print(R_M[20:-1,20:-1])
    #print(R_M[0:10,0:10])
    # plt.imshow(R_M)
    # plt.colorbar()
    # plt.show()

    # ONLY FOR GRADIENT DESCENT
    # # Matrix B_{(B),i} dim = D x D*m
    # B_B = []
    # for i in range(n):
    #     BB_temp = beta[0][i]*np.eye(D)    
    #     for j in range(1,m):
    #         BB_temp = np.concatenate((BB_temp, beta[j][i]*np.eye(D)),axis=-1)
    #     B_B.append(BB_temp)

    # #print(B_B[0].shape)

    # # Observation-specific coefficient matrix B_{i} dim = (nbins+1)*D x (nbins+1)*D*m
    # B = []
    # for i in range(n):
    #     # create first row block
    #     B_temp = np.concatenate((B_B[i],np.zeros(((B_B[i].shape[0]),(nbins+1)*D*m-D*m))),axis=-1)  
    #     # create other rows
    #     for j in range(1,nbins+1):
    #         B_temp_row = np.zeros((D,D*m))
    #         for s in range(1,nbins+1):
    #             if j == s:
    #                 B_temp_row = np.concatenate((B_temp_row, B_B[i]), axis=-1)
    #             else:
    #                 B_temp_row = np.concatenate((B_temp_row, np.zeros((D,D*m))), axis=-1)
    #         B_temp = np.concatenate((B_temp,B_temp_row), axis=0)
            
    #     B.append(B_temp)

    # #print(B[0].shape)
    # # plt.imshow(B[0])
    # # plt.colorbar()
    # # plt.show()

    return W, C_M, R_M

def basis(x, mu, bases, beta, W_bn, n, nbins, D, m, w_l, w_u, theta, theta_low, theta_up):
    # Function that computes the matrices necessary for basis update
    
    # Matrix R_{(V)} dim = D*m*(nbins+1) x D*m*(nbins+1)
    R_V = 2*np.eye(D*m*(nbins+1))

    for i in range(D*m):    # first and last block
        R_V[i][i] = 1
        R_V[D*m*(nbins+1)-1-i][D*m*(nbins+1)-1-i] = 1 

    for i in range(1,nbins+1):
        for j in range(D*m):
            R_V[(i-1)*D*m+j][i*D*m+j] = -1
            R_V[i*D*m+j][(i-1)*D*m+j] = -1

    # print(R_V.shape)
    # plt.imshow(R_V)
    # plt.colorbar()
    # plt.show()

    # Vector b_{i} dim = (nbins+1)*D*m
    b_i = []
    for i in range(n):
        b_temp = []
        P = compute.compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)
        temp_vec = x[:3,i]-mu[:,i]-(P[:,0]*beta[i,0])-(P[:,1]*beta[i,1])
        for j in range(nbins+1):
            for s in range(m):
                temp_prod = W_bn[j,i]*beta[i,s]
                b_temp.append(temp_prod*temp_vec)

        b_i.append(np.stack(b_temp).reshape((1,-1)))

    #print(np.stack(b_i).shape)

    return R_V, b_i

def T_bvw(b,v,w,D,m,nbins):
    # Transition-like matrix T_{b,v,w} dim = D*m*(nbins+1) x D*m*(nbins+1)
    T_temp = np.zeros((D*m*(nbins+1),D*m*(nbins+1)))

    for i in range(D):
        T_temp[D*(b*m+v)+i][D*(b*m+w)+i] = 1
    
    # test = T_bvw(3,0,0,3,2,4)
    # print(test.shape)
    # plt.imshow(test)
    # plt.colorbar()
    # plt.show()
    return T_temp
