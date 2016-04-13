from __future__ import division

import numpy as np
import numpy.random as npr

from utils import mvn_weighted_logged, sample_discrete, stick_break_proc
from wishart import invwishartrand_prec
from munkres import munkres, get_cost

# noinspection PyUnresolvedReferences, PyPackageRequirements
import sampler


# check for GPU compatibility
try:
    # noinspection PyPackageRequirements, PyUnresolvedReferences
    import pycuda
    # noinspection PyPackageRequirements, PyUnresolvedReferences
    import pycuda.driver
    # noinspection PyUnresolvedReferences
    try:
        from utils_gpu import init_GPUWorkers, get_labelsGPU
        _has_gpu = True
    except (ImportError, pycuda.driver.RuntimeError):
        _has_gpu = False
except ImportError:
    _has_gpu = False


class DPNormalMixture(object):
    """
    MCMC sampling for Truncated Dirichlet Process Mixture of Normals

    Parameters
    ----------
    data : ndarray (nobs x ndim) or (BEM_)DPNormalMixture class
    ncomp : int
        Number of mixture components

    Notes
    -----
    y ~ \sum_{j=1}^J \pi_j {\cal N}(\mu_j, \Sigma_j)
    \pi ~ stickbreak(\alpha)
    \alpha ~ Ga(e, f)
    \mu_j ~ N(0, m\Sigma_j)
    \Sigma_j ~ IW(nu0 + 2, nu0 * \Phi_j)

    The defaults for the prior parameters are reasonable for
    standardized data. However, a careful analysis should always
    include careful choosing of priors.

    Citations
    ---------

    Ishwaran and James.
    'Gibbs Sampling Methods for Stick-Breaking Priors'
    (2001)
    
    M. Suchard, Q. Wang, C. Chan, J. Frelinger, A. Cron and M. West.
    'Understanding GPU programming for statistical computation:
    Studies in massively parallel massive mixtures.'
    Journal of Computational and Graphical Statistics.
    19 (2010): 419-438
    """
    def __init__(self, data, ncomp=256, gamma0=10, m0=None,
                 nu0=None, Phi0=None, e0=10, f0=1,
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 parallel=True, verbose=False):

        # regardless of data class or _has_gpu, initialize gpu data to None
        # this gets set in sample method if a gpu device is available
        self.gpu_data = None

        if issubclass(type(data), DPNormalMixture):
            self.data = data.data
            self.nobs, self.ndim = self.data.shape
            self.ncomp = data.ncomp
            self.mu_prior_mean = data.mu_prior_mean
            nu0 = data._nu0
            Phi0 = data._Phi0
            if len(data.mu.shape) > 2:
                mu0 = data.mu[-1].copy()
                Sigma0 = data.Sigma[-1].copy()
                weights0 = data.weights[-1].copy()
            else:
                mu0 = data.mu.copy()
                Sigma0 = data.Sigma.copy()
                weights0 = data.weights.copy()
            e0 = data.e
            f0 = data.f
            self.gamma = data.gamma
            self.parallel = data.parallel
        else:
            self.data = np.asarray(data)
            self.nobs, self.ndim = self.data.shape
            self.ncomp = ncomp

            # TODO hyper-parameters
            # prior mean for component means
            if m0 is not None:
                if len(m0) == self.ndim:
                    self.mu_prior_mean = m0.copy()
                elif len(m0) == 1:
                    self.mu_prior_mean = m0 * np.ones(self.ndim)
            else:
                self.mu_prior_mean = np.zeros(self.ndim)

            self.gamma = gamma0*np.ones(ncomp)
            self.parallel = parallel
                        
        self.verbose = verbose

        self._set_initial_values(
            alpha0,
            nu0,
            Phi0,
            mu0,
            Sigma0,
            weights0,
            e0,
            f0
        )

        # Check data for non-contiguous crap
        if not (self.data.flags["C_CONTIGUOUS"] or
                self.data.flags["F_CONTIGUOUS"]):
            self.data = self.data.copy()
        
    def _set_initial_values(self, alpha0, nu0, Phi0, mu0, Sigma0, weights0,
                            e0, f0):
        if nu0 is None:
            nu0 = 1

        if Phi0 is None:
            Phi0 = np.empty((self.ncomp, self.ndim, self.ndim))
            Phi0[:] = np.eye(self.ndim) * nu0

        if Sigma0 is None:
            # draw from prior .. bad idea for vague prior ??? 
            Sigma0 = np.empty((self.ncomp, self.ndim, self.ndim))
            for j in xrange(self.ncomp):
                Sigma0[j] = invwishartrand_prec(nu0 + 1 + self.ndim, Phi0[j])

        # starting values, are these sensible?
        if mu0 is None:
            mu0 = np.empty((self.ncomp, self.ndim))
            for j in xrange(self.ncomp):
                mu0[j] = npr.multivariate_normal(
                    np.zeros(self.ndim),
                    self.gamma[j] * Sigma0[j]
                )

        if weights0 is None:
            weights0 = (1/self.ncomp)*np.ones((self.ncomp, 1))

        self._alpha0 = alpha0
        self.e = e0
        self.f = f0

        self._weights0 = weights0
        self._mu0 = mu0
        self._Sigma0 = Sigma0
        self._nu0 = nu0  # prior degrees of freedom
        self._Phi0 = Phi0  # prior location for Sigma_j's

    def sample(
            self,
            niter=1000,
            nburn=0,
            thin=1,
            ident=False,
            device=None,
            callback=None
    ):
        """
        samples niter + nburn iterations only storing the last niter
        draws thinned as indicated.

        if ident is True the munkres identification algorithm will be
        used matching to the INITIAL VALUES. These should be selected
        with great care. We recommend using the EM algorithm. Also
        .. burn-in doesn't make much sense in this case.

        Parameters
        ----------
        niter
        nburn
        thin
        ident
        device
        callback

        Returns
        -------
        None
        """

        if _has_gpu and device is not None:
            # if a gpu is available, send data to device & save gpu_data
            self.gpu_data = init_GPUWorkers(self.data, device)
            if self.verbose:
                print "starting GPU enabled MCMC"
        else:
            if self.verbose:
                print "starting MCMC"

        self._setup_storage(niter)

        alpha = self._alpha0
        weights = self._weights0
        mu = self._mu0
        sigma = self._Sigma0

        for i in range(-nburn, niter):
            if isinstance(self.verbose, int) and self.verbose and \
                    not isinstance(self.verbose, bool):
                if i % self.verbose == 0:
                    print i

            if hasattr(callback, '__call__'):
                callback(i)

            # update labels
            labels, z_hat = self._update_labels(
                mu,
                sigma,
                weights,
                ident=ident
            )

            # get initial reference if needed
            if i == 0 and ident:
                z_ref = z_hat.copy()
                c0 = np.zeros((self.ncomp, self.ncomp), dtype=np.double)
                for j in xrange(self.ncomp):
                    # noinspection PyTypeChecker
                    c0[j, :] = np.sum(z_ref == j)

            # update mu and sigma
            counts = self._update_mu_sigma(mu, sigma, labels)

            # update weights
            stick_weights, weights = self._update_stick_weights(counts, alpha)

            alpha = self._update_alpha(stick_weights)

            # relabel if needed:
            if i > 0 and ident:
                cost = c0.copy()
                get_cost(z_ref, z_hat, cost)

                _, iii = np.where(munkres(cost))
                weights = weights[iii]
                mu = mu[iii]
                sigma = sigma[iii]

            # save
            if i >= 0:
                self.weights[i] = weights
                self.alpha[i] = alpha
                self.mu[i] = mu
                self.Sigma[i] = sigma

    def _setup_storage(self, niter=1000, thin=1):
        n_results = niter // thin

        self.weights = np.zeros((n_results, self.ncomp))
        self.mu = np.zeros((n_results, self.ncomp, self.ndim))
        self.Sigma = np.zeros((n_results, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.zeros(n_results)

    def _update_labels(self, mu, Sigma, weights, ident=False):
        if self.gpu_data is not None:
            return get_labelsGPU(
                self.gpu_data,
                weights,
                mu,
                Sigma,
                relabel=ident
            )
        else:
            densities = mvn_weighted_logged(self.data, mu, Sigma, weights)
            if ident:
                z = np.asarray(densities.argmax(1), dtype='i')
            else:
                z = None
            return sample_discrete(densities).squeeze(), z

    def _update_stick_weights(self, counts, alpha):

        reverse_cum_sum = counts[::-1].cumsum()[::-1]

        a = 1 + counts[:-1]
        b = alpha + reverse_cum_sum[1:]
        stick_weights, mixture_weights = stick_break_proc(a, b)
        return stick_weights, mixture_weights

    def _update_alpha(self, v):
        # is v the current stick weights?
        a = self.ncomp + self.e - 1
        b = self.f - np.log(1 - v).sum()
        return npr.gamma(a, scale=1 / b)

    def _update_mu_sigma(self, mu, sigma, labels):
        counts = sampler.sample_mu_Sigma(
            mu,
            sigma,
            np.asarray(labels, dtype=np.int),
            self.data,
            self.gamma[0],
            self.mu_prior_mean,
            self._nu0,
            self._Phi0[0],
            self.parallel
        )
        return counts
