from weights import setting_weights
from weights import get_bounds
from compute import initialization
from compute import compute_P, compute_mu
import numpy as np
import scipy
from sklearn.preprocessing import normalize

def compute_losses(x, mu_len, p_len, xdata, w_l, w_u, theta, theta_low, theta_up, lambda_v, lambda_0, m):
    E_data = 0
    nbins = len(theta)-1
    n, D = xdata.shape
    bases = x.reshape(((nbins+1)*m,D)).reshape((nbins+1,m,D))

    betas = update_betas(n, xdata, w_l, w_u, np.zeros((nbins+1,D)), bases, theta, theta_low, theta_up)

    for i in range(n):
        p_new = []
        for j in range(m):
            p_new.append(compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)[:,j]/np.linalg.norm(compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)[:,j]))
        res = xdata[i]- np.vstack(p_new).T @ betas[i]
        E_data += (np.linalg.norm(res)**2)
    E_data = E_data/n

    E_smo = 0
    res_mu = 0
    res_bas = 0
    E_ort = 0
    for b in range(nbins):
        for v in range(m):
            b1 = bases[b][v]/np.linalg.norm(bases[b][v])
            b2 = bases[b+1][v]/np.linalg.norm(bases[b][v])
            res_bas += np.linalg.norm(b1-b2)**2
            for w in np.arange(v,m):
                if w == v:
                    ind = 1
                else:
                    ind = 0
                bv = bases[b][v]/np.linalg.norm(bases[b][v])
                bw = bases[b][w]/np.linalg.norm(bases[b][w])
                E_ort += (np.dot(bv,bw)-ind)**2
    E_smo = lambda_v/nbins*(res_bas)
    E_ort = lambda_0*E_ort

    return E_data + E_smo + E_ort

def update_betas(n, xdata, w_l, w_u, mu_prec, bases, theta, theta_low, theta_up):
  update = []
  for i in range(n):
      P = compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)
      res = (xdata[i]-compute_mu(w_l, w_u, mu_prec, theta, theta_low, theta_up, i))
      update.append(np.linalg.lstsq(P, res, rcond = None)[0])

  update = np.stack(update)
  return update

class GuptaPPCA(object):

  def __init__(self, bins, m):
    self.bins = bins
    self.m = m
    self.dim = None

  def fit(self, thetadata, xdata, lambda_v = 4.2, lambda_m = 0.008, lambda_O = 20, method='Nelder-Mead', tol = 0.005, maxiter = 100, verbose = False):
    ''' thetadata shape (N,), xdata shape (N, dim)'''
    n, D = xdata.shape
    self.dim = D
    bins = self.bins
    m = self.m
    nbins = len(bins)-1
    lambda_0 = lambda_O

    theta_up, theta_low, w_l, w_u, w_bn = setting_weights(n, nbins, thetadata, bins)

    mu_0, bases_0, betas_0 = initialization(n, nbins, m, xdata.T, w_bn, w_l, w_u, bins, theta_low, theta_up)
    mean_0 = mu_0.reshape(((nbins+1)*D,))
    p_0 = bases_0.reshape(((nbins+1)*m*D,))

    mu_len = mean_0.shape[0]
    p_len = p_0.shape[0]
    mu_history = [mu_0]
    bases_history = [bases_0]
    betas_history = [betas_0]

    # Compute E_{0}
    E_0 = compute_losses(p_0, mu_len, p_len, xdata, w_l, w_u, bins, theta_low, theta_up, lambda_v, lambda_0, m)
    E_history = [E_0]

    # Optimization
    res = scipy.optimize.minimize(compute_losses, p_0, args=(mu_len, p_len, xdata, w_l, w_u, bins, theta_low, theta_up, lambda_v, lambda_0, m), method=method, tol=tol, options = {'maxiter':maxiter,'disp':verbose})

    # Mu
    self.mu = np.zeros((nbins+1,D))

    # P
    self.bases = res.x.reshape(((nbins+1)*m,D)).reshape((nbins+1,m,D))

    # Betas
    self.betas = update_betas(n, xdata, w_l, w_u, self.mu, self.bases, bins, theta_low, theta_up)

    # Normalize bases
    for b in range(nbins+1):
        self.bases[b] = normalize(self.bases[b], axis=1)

  def predict(self, theta):
    bins = self.bins
    nbins = len(bins)-1
    theta_up_new, theta_low_new, w_l_new, w_u_new, _ = setting_weights(1, nbins, [theta], bins)
    return compute_P(w_l_new, w_u_new, self.bases, bins, theta_low_new, theta_up_new, 0)
  
  def transform(self, thetadata, xdata):
    ''' thetadata shape (N,), xdata shape (N, dim)'''
    xdata_proj = np.zeros((len(xdata),self.m))

    for i in range(len(xdata)):
        # xdata_proj[i,:] = xdata[i,:] @ self.predict(thetadata[i])
        xdata_proj[i,:] = np.linalg.lstsq(self.predict(thetadata[i]), xdata[i,:], rcond = None)[0]

    return xdata_proj

  def inverse_transform(self, thetadata, xdata_proj):
    ''' thetadata shape (N,), xdata_proj shape (N, modes)'''
    xdata_reconstruction = np.zeros((len(xdata_proj),self.dim))

    for i in range(len(xdata_proj)):
        xdata_reconstruction[i,:] = xdata_proj[i,:] @ self.predict(thetadata[i]).T
    
    return xdata_reconstruction
  
class GrassPPCA(object):
    ''' Class that implements PCA interpolation '''
    def __init__(self, bins, m):
        self.bins = bins
        self.m = m
        self.b_bases = []
        self.dim = None

    def create_dataset(self,endpoint,thetadata,xdata):
        train_grass = []
        n = len(xdata)
        # create train for first bin endpoint
        if endpoint == 0:
            low_mid = self.bins[endpoint]
            up_mid = 0.5*(self.bins[endpoint+1]+self.bins[endpoint])
            index = (thetadata<=up_mid)*(thetadata > low_mid)
            if sum(index)<10:
                up_mid = self.bins[endpoint+1]
                index = (thetadata<=up_mid)*(thetadata > low_mid)
        # create train for last bin endpoint
        elif endpoint == len(self.bins)-1:
            low_mid = 0.5*(self.bins[endpoint-1]+self.bins[endpoint])
            up_mid = self.bins[endpoint]
            index = (thetadata<=up_mid)*(thetadata > low_mid)
            if sum(index)<10:
                low_mid = self.bins[endpoint-1]
                index = (thetadata<=up_mid)*(thetadata > low_mid)
        
        else:
            # create train for middle bin endpoints
            low_mid = 0.5*(self.bins[endpoint-1]+self.bins[endpoint])
            up_mid = 0.5*(self.bins[endpoint+1]+self.bins[endpoint])
            index = (thetadata<=up_mid)*(thetadata > low_mid)
        train_grass = xdata[index]

        return train_grass

    def fit(self, thetadata, xdata):
        ''' thetadata shape (N,), xdata shape (N, dim)'''
        b_bases = []
        self.dim = xdata.shape[-1]
        for b in range(len(self.bins)):
            data_b = self.create_dataset(b,thetadata,xdata)
            cov = data_b.T@data_b/(len(data_b)-1)
            eigval, eigvec = np.linalg.eigh(cov)

            # Sorting eigenvals and eigvectors based on eigvals
            idx = eigval.argsort()[::-1]   
            eigval = eigval[idx]
            eigvec = eigvec[:,idx]

            b_bases.append(eigvec[:,:self.m])

        self.b_bases = b_bases

    def predict(self,theta_new):
        # Function that implements base interpolation
        if theta_new == self.bins[-1]:
            s0 = self.bins[-2]
            s1 = self.bins[-1]
            A = self.b_bases[-2]
            B = self.b_bases[-1]
        else:
            i_l = max(np.where(self.bins<=theta_new)[0])
            s0 = self.bins[i_l]
            s1 = self.bins[i_l+1]
            A = self.b_bases[i_l]
            B = self.b_bases[i_l+1]
        inv_AB = np.linalg.inv(A.T@B+1e-10)
        mult = (np.eye(A.shape[0])-A@A.T)@B@inv_AB
        U, S, Vh = np.linalg.svd(mult, full_matrices=False)
        return A @ Vh.T @ np.diag(np.cos((theta_new-s0)/(s1-s0)*np.arctan(S))) + U @ np.diag(np.sin((theta_new-s0)/(s1-s0)*np.arctan(S)))
    
    def transform(self, thetadata, xdata):
        ''' thetadata shape (N,), xdata shape (N, dim)'''
        xdata_proj = np.zeros((len(xdata),self.m))

        for i in range(len(xdata)):
            xdata_proj[i,:] = xdata[i,:] @ self.predict(thetadata[i])

        return xdata_proj
    
    def inverse_transform(self, thetadata, xdata_proj):
        ''' thetadata shape (N,), xdata_proj shape (N, modes)'''
        xdata_reconstruction = np.zeros((len(xdata_proj),self.dim))

        for i in range(len(xdata_proj)):
            xdata_reconstruction[i,:] = xdata_proj[i,:] @ self.predict(thetadata[i]).T
        
        return xdata_reconstruction
    
class KernelPPCA(object):

    def __init__(self, thetadata, xdata, m, kernel):
        ''' thetadata shape (N,), xdata shape (N, dim)'''
        self.thetadata = thetadata
        self.xdata = xdata
        self.dim = xdata.shape[-1]
        self.m = m
        self.kernel = kernel

    def scale_dataset(self, theta_new):

        new_data = np.zeros(self.xdata.shape)
        for i in range(len(self.xdata)):
            new_data[i,:] = self.xdata[i,:]*self.kernel(theta_new,self.thetadata[i])
        
        return new_data
    
    def predict(self, theta_new):
        D = self.scale_dataset(theta_new)
        cov = D.T@D/(len(D)-1)
        eigval, eigvec = np.linalg.eigh(cov)

        # Sorting eigenvals and eigvectors based on eigvals
        idx = eigval.argsort()[::-1]   
        eigval = eigval[idx]
        eigvec = eigvec[:,idx]

        return eigvec[:,:self.m]
    
    def transform(self, thetadata, xdata):
        ''' thetadata shape (N,), xdata shape (N, dim)'''
        xdata_proj = np.zeros((len(xdata),self.m))
        for i in range(len(xdata)):
            xdata_proj[i,:] = xdata[i,:] @ self.predict(thetadata[i])

        return xdata_proj
    
    def inverse_transform(self, thetadata, xdata_proj):
        ''' thetadata shape (N,), xdata_proj shape (N, modes)'''    
        xdata_reconstruction = np.zeros((len(xdata_proj),self.dim))

        for i in range(len(xdata_proj)):
            xdata_reconstruction[i,:] = xdata_proj[i,:] @ self.predict(thetadata[i]).T
        
        return xdata_reconstruction
    
class binPCA(object):
    ''' Class that implements PCA independently for each bin '''
    def __init__(self, bins, m):
        self.bins = bins
        self.m = m
        self.bin_bases = []
        self.dim = None

    def fit(self, thetadata, xdata):
        ''' thetadata shape (N,), xdata shape (N, dim)'''
        theta_low = get_bounds(thetadata, self.bins)[-1]
        self.dim = xdata.shape[-1]
        bin_bases = []

        for b in range(len(self.bins)-1):
            i_bin = np.where(theta_low==self.bins[b])[0]
            data_bin = xdata[i_bin,:]
            cov = data_bin.T@data_bin/(len(data_bin)-1)
            eigval, eigvec = np.linalg.eigh(cov)

            # Sorting eigenvals and eigvectors based on eigvals
            idx = eigval.argsort()[::-1]   
            eigval = eigval[idx]
            eigvec = eigvec[:,idx]

            bin_bases.append(eigvec[:,:self.m])

        self.bin_bases = bin_bases

    def predict(self,new_theta):
        return self.bin_bases[sum(self.bins[:-1]<=new_theta)-1]
    
    def transform(self, thetadata, xdata):
        ''' thetadata shape (N,), xdata shape (N, dim)'''
        theta_low = get_bounds(thetadata, self.bins)[-1]
        xdata_proj = np.zeros((len(xdata),self.m))

        for b in range(len(self.bins)-1):
            i_bin = np.where(theta_low==self.bins[b])[0]
            xdata_proj[i_bin,:] = xdata[i_bin,:] @ self.bin_bases[b]

        return xdata_proj
    
    def inverse_transform(self, thetadata, xdata_proj):
        ''' thetadata shape (N,), xdata_proj shape (N, modes)'''
        theta_low = get_bounds(thetadata, self.bins)[-1]
        xdata_reconstruction = np.zeros((len(xdata_proj),self.dim))

        for b in range(len(self.bins)-1):
            i_bin = np.where(theta_low==self.bins[b])[0]
            xdata_reconstruction[i_bin,:] = xdata_proj[i_bin,:] @ self.bin_bases[b].T
        
        return xdata_reconstruction
