import numpy as np
from sklearn.decomposition import PCA
import copy

def compute_mu(w_l, w_u, mu_bin, theta, theta_low, theta_up, index):
    # Function that computes the mean given theta_i
    i_low = np.where(theta==theta_low[index])
    i_up = np.where(theta==theta_up[index])
    return np.squeeze(w_l[index]*mu_bin[i_low] + w_u[index]*mu_bin[i_up])

def compute_P(w_l, w_u, bases, theta, theta_low, theta_up, index):
    # Function that computes the matrix P given theta_i
    i_low = np.where(theta==theta_low[index])
    i_up = np.where(theta==theta_up[index])
    res = np.squeeze(w_l[index]*bases[i_low] + w_u[index]*bases[i_up])
    return res.T

def initialization(n, nbins, m, data, w_bn, w_l, w_u, theta, theta_low, theta_up, eps = 0.001):
    # Function that initializes mean, bases and coeff.
    mu_0 = []
    bases = []
    betas = []

    for b in range(nbins+1):
        mu_0.append((np.matmul(data[:3,:],w_bn[b,:]))/(w_bn[b,:].sum()))
        # print(np.shape(mu_0))
         
        # Perform PCA on examples with w_{b,i} > eps
        indices = np.where(w_bn[b,:]>eps)
        train_init = data[:3,indices].reshape((3,np.shape(indices)[-1])).T
        pca = PCA(n_components=m)
        pca.fit(train_init)
        p1 = pca.components_[0]
        p2 = pca.components_[1]

        bases.append(np.vstack((p1,p2)))
        #print(bases[0].shape)
        #print(pca.explained_variance_ratio_)

    # rearrange vectors
    for b in range(1,nbins+1):
        prod1 = np.dot(bases[b-1][0],bases[b][0]) 
        prod2 = np.dot(bases[b-1][0],bases[b][1])
        if abs(prod1) < abs(prod2):
            temp = copy.copy(bases[b][0])
            bases[b][0] = bases[b][1]
            bases[b][1] = temp

        if np.dot(bases[b-1][0],bases[b][0]) < 0:
            bases[b][0] = -bases[b][0]
        if np.dot(bases[b-1][1],bases[b][1]) < 0:
            bases[b][1] = -bases[b][1]

    mu_0 = np.stack(mu_0)
    bases = np.stack(bases)

    for i in range(n):
        P = compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)
        P_inv = np.linalg.pinv(P)
        res = (data[:3,i]-compute_mu(w_l, w_u, mu_0, theta, theta_low, theta_up, i))

        betas.append(P_inv @ res)

    betas = np.stack(betas)

    return mu_0, bases, betas