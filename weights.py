import numpy as np

def get_bounds(x,bins):
    # Function to get upper and lower bounds for each observation    
    up = np.zeros(len(x))
    low = np.zeros(len(x))
    for i in range(len(x)):
        j = 0
        flag = True
        while(flag):
            if x[i]<=bins[j]:
                up[i] = bins[j]
                low[i] = bins[j-1]
                flag = False
                
            j += 1 

    return up,low   
def setting_weights(n, nbins, data, theta):

    theta_up, theta_low = get_bounds(data[3][:],theta)
    #print(theta_up[:5])
    #print(theta_low[:5])

    # Computing the weights
    w_l = (theta_up-data[3][:])/(theta_up-theta_low)
    w_u = (data[3][:]-theta_low)/(theta_up-theta_low)

    #print(w_l[:3])
    #print(w_u[:3])

    # Bin endpoints weights  w_{b,i}
    w_bn = np.zeros((nbins+1,n))
    for i in range(n):
        for j in range(nbins+1):
            if theta[j] == theta_low[i]:
                w_bn[j][i] = w_l[i]
            if theta[j] == theta_up[i]:
                w_bn[j][i] = w_u[i]

    #print(w_bn[:,0])
    return theta_up, theta_low, w_l, w_u, w_bn