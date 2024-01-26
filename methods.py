from weights import setting_weights
from compute import initialization
from compute import compute_P, compute_mu
import numpy as np
import scipy
from sklearn.preprocessing import normalize

def compute_losses(x, mu_len, p_len, xdata, w_l, w_u, theta, theta_low, theta_up, lambda_v, lambda_0, m):
    E_data = 0
    nbins = len(theta)-1
    n, K = xdata.shape
    bases = x.reshape(((nbins+1)*m,K)).reshape((nbins+1,m,K))

    betas = update_betas(n, xdata, w_l, w_u, np.zeros((nbins+1,K)), bases, theta, theta_low, theta_up)

    for i in range(n):
        p1_new = compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)[:,0]/np.linalg.norm(compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)[:,0])
        p2_new = compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)[:,1]/np.linalg.norm(compute_P(w_l, w_u, bases, theta, theta_low, theta_up, i)[:,1])
        res = xdata[i]- np.vstack((p1_new,p2_new)).T @ betas[i]
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
  
  def fit(self, thetadata, xdata, lambda_v = 0.008, lambda_m = 4.2, lambda_O = 20, tol = 0.005, maxiter = 100, verbose = False):
    ''' thetadata shape (N,), xdata shape (N, dim)'''
    n, K = xdata.shape
    bins = self.bins
    m = self.m
    nbins = len(bins)-1
    lambda_0 = lambda_O

    theta_up, theta_low, w_l, w_u, w_bn = setting_weights(n, nbins, thetadata, bins)

    mu_0, bases_0, betas_0 = initialization(n, nbins, m, xdata.T, w_bn, w_l, w_u, bins, theta_low, theta_up)
    mean_0 = mu_0.reshape(((nbins+1)*K,))
    p_0 = bases_0.reshape(((nbins+1)*m*K,))

    mu_len = mean_0.shape[0]
    p_len = p_0.shape[0]
    mu_history = [mu_0]
    bases_history = [bases_0]
    betas_history = [betas_0]

    # Compute E_{0}
    E_0 = compute_losses(p_0, mu_len, p_len, xdata, w_l, w_u, bins, theta_low, theta_up, lambda_v, lambda_0, m)
    E_history = [E_0]

    # Optimization
    res = scipy.optimize.minimize(compute_losses, p_0, args=(mu_len, p_len, xdata, w_l, w_u, bins, theta_low, theta_up, lambda_v, lambda_0, m), method='Nelder-Mead', tol=tol, options = {'maxiter':maxiter,'disp':verbose})

    # Mu
    self.mu = np.zeros((nbins+1,K))

    # P
    self.bases = res.x.reshape(((nbins+1)*m,K)).reshape((nbins+1,m,K))

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
