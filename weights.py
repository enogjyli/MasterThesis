import numpy as np

def get_bounds(x,bins):
    # Function to get upper and lower bounds for each observation    
    up = np.zeros(len(x))
    low = np.zeros(len(x))
    for i in range(len(x)):
        j = 1
        flag = True
        while(flag):
            if x[i]<=bins[j]:
                up[i] = bins[j]
                low[i] = bins[j-1]
                flag = False
                
            j += 1 

    return up,low   
def setting_weights(n, nbins, thetadata, theta):

    theta_up, theta_low = get_bounds(thetadata,theta)

    # Computing the weights
    w_l = (theta_up-thetadata)/(theta_up-theta_low)
    w_u = (thetadata-theta_low)/(theta_up-theta_low)

    # Bin endpoints weights  w_{b,i}
    w_bn = np.zeros((nbins+1,n))
    for i in range(n):
        for j in range(nbins+1):
            if theta[j] == theta_low[i]:
                w_bn[j][i] = w_l[i]
            if theta[j] == theta_up[i]:
                w_bn[j][i] = w_u[i]

    return theta_up, theta_low, w_l, w_u, w_bn
