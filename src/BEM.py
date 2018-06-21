"""
Created on Mar 15, 2012

@author: Andrew Cron
"""
import numpy as np

from .utils import mvn_weighted_logged
from .dpmix import DPNormalMixture

# check for GPU compatibility
try:
    # noinspection PyPackageRequirements, PyUnresolvedReferences
    import pycuda
    # noinspection PyPackageRequirements, PyUnresolvedReferences
    import pycuda.driver
    # noinspection PyUnresolvedReferences
    try:
        from .utils_gpu import init_gpu_data, get_bem_densities_gpu
        _has_gpu = True
    except (ImportError, pycuda.driver.RuntimeError):
        _has_gpu = False
except ImportError:
    _has_gpu = False


class BEMNormalMixture(DPNormalMixture):
    """
    BEM algorithm for finding the posterior mode of the
    Truncated Dirichlet Process Mixture of Models 

    Parameters
    ----------
    data : ndarray (nobs x ndim)
    ncomp : int
        Number of mixture components

    Notes
    -----
    y ~ \sum_{j=1}^J \pi_j {\cal N}(\mu_j, \Sigma_j)
    \alpha ~ Ga(e, f)
    \Sigma_j ~ IW(nu0 + 2, nu0 * \Phi_j)

    Citation
    --------

    M. Suchard, Q. Wang, C. Chan, J. Frelinger, A. Cron and
    M. West. 'Understanding GPU programming for statistical
    computation: Studies in massively parallel massive mixtures.'
    Journal of Computational and Graphical Statistics. 19 (2010):
    419-438
    """
    def __init__(self, data, ncomp=256, gamma0=100, m0=None,
                 nu0=None, Phi0=None, e0=10, f0=1,
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 parallel=True, verbose=False):

        parallel = False  # Need to cythonize ....

        # for now, initialization is exactly the same ....
        super(BEMNormalMixture, self).__init__(
            data, ncomp, gamma0, m0, nu0, Phi0, e0, f0,
            mu0, Sigma0, weights0, alpha0, parallel, verbose)

        self.alpha = self._alpha0
        self.weights = self._weights0.copy()
        self.stick_weights = self.weights.copy()
        self.mu = self._mu0.copy()
        self.Sigma = self._Sigma0.copy()
        self.e_labels = np.tile(self.weights.flatten(), (self.nobs, 1))
        self.densities = None
        self.log_likelihood = None
        self.ct = None
        self.x_bar = None

    def optimize(self, maxiter=1000, perdiff=0.1, device=None):
        """
        Optimizes the posterior distribution given the data. The
        algorithm terminates when either the maximum number of
        iterations is reached or the percent difference in the
        posterior is less than perdiff.
        """
        if _has_gpu and device is not None:
            if self.verbose:
                print("starting GPU enabled BEM")
            self.gpu_data = init_gpu_data(self.data, device)
        else:
            if self.verbose:
                print("starting BEM")

        self.expected_labels()
        ll_2 = self.log_posterior()
        ll_1 = 1
        it = 0

        while np.abs(ll_1 - ll_2) > 0.01 * perdiff and it < maxiter:
            it += 1

            self.maximize_mu()
            self.maximize_sigma()
            self.maximize_weights()
            self.expected_alpha()
            self.expected_labels()
            ll_1 = ll_2
            ll_2 = self.log_posterior()
                
    def log_posterior(self):
        # just the log likelihood right now because im lazy ... 
        return self.log_likelihood

    def expected_labels(self):
        if self.gpu_data is not None:
            densities = get_bem_densities_gpu(
                self.gpu_data,
                self.weights,
                self.mu,
                self.Sigma
            )

            densities = np.exp(densities)
            norm = densities.sum(1)
            self.log_likelihood = np.sum(np.log(norm))
            densities = (densities.T / norm).T

            self.ct = np.asarray(densities.sum(0), dtype='d')
            self.x_bar = np.asarray(
                np.dot(densities.T, self.data),
                dtype='d'
            )
            self.densities = densities.copy('C')

        else:
            densities = mvn_weighted_logged(
                self.data,
                self.mu,
                self.Sigma,
                self.weights
            )
            densities = np.exp(densities)
            norm = densities.sum(1)
            self.log_likelihood = np.sum(np.log(norm))
            densities = (densities.T / norm).T
            self.ct = densities.sum(0)
            self.x_bar = np.dot(densities.T, self.data)
            self.densities = densities

    def expected_alpha(self):
        sm = np.sum(np.log(1. - self.stick_weights[:-1]))
        self.alpha = (self.ncomp + self.e - 1.) / (self.f - sm)

    def maximize_mu(self):
        k, p = self.ncomp, self.ndim
        self.mu = \
            (
                np.tile(self.mu_prior_mean, (k, 1)) +
                np.tile(self.gamma.reshape(k, 1), (1, p)) * self.x_bar
            ) / np.tile((1. + self.gamma * self.ct).reshape(k, 1), (1, p))

    def maximize_sigma(self):
        for j in range(self.ncomp):
            if self.ct[j] > 0.1:
                xj_d = (self.data - self.x_bar[j, :] / self.ct[j])
                ss = np.dot(xj_d.T * self.densities[:, j].flatten(), xj_d)
                ss += self._Phi0[j] + \
                    (self.ct[j] / (1 + self.gamma[j] * self.ct[j])) * np.outer(
                        (1/self.ct[j]) * self.x_bar[j, :] - self.mu_prior_mean,
                        (1/self.ct[j]) * self.x_bar[j, :] - self.mu_prior_mean
                    )
                self.Sigma[j] = ss / self.ct[j]

    def maximize_weights(self):
        self.stick_weights = np.minimum(
            np.ones(len(self.stick_weights)) - 1e-10,
            self.ct / (self.alpha - 1 + self.ct[::-1].cumsum()[::-1])
        )
        self.stick_weights = np.maximum(
            np.ones(len(self.stick_weights)) * 1e-10,
            self.stick_weights
        )
        self.stick_weights[-1] = 1.

        v = self.stick_weights[:-1]
        pi = self.weights
        
        pi[0] = v[0]
        prod = (1 - v[0])
        for k in range(1, len(v)):
            pi[k] = prod * v[k]
            prod *= 1 - v[k]
        pi[-1] = prod
