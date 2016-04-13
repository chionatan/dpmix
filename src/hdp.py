from __future__ import division

import numpy as np
from scipy import stats

from utils import mvn_weighted_logged, sample_discrete, stick_break_proc, \
    break_sticks
from dpmix import DPNormalMixture
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
        from utils_gpu import init_GPUWorkers, get_hdp_labels_GPU
        _has_gpu = True
    except (ImportError, pycuda.driver.RuntimeError):
        _has_gpu = False
except ImportError:
    _has_gpu = False


class HDPNormalMixture(DPNormalMixture):
    """
    MCMC sampling for Doubly Truncated HDP Mixture of Normals for multiple
    data sets

    Parameters
    -----------
    data :  list of ndarrays (nobs x ndim) -- ndim must be equal .. not nobs
    ncomp : nit
        Number of mixture components

    Notes
    -----
    y_j ~ \sum_{k=1}^K \pi_{kj} {\cal N}(\mu_k, \Sigma_k)
    \beta ~ stickbreak(\alpha)
    \alpha ~ Ga(e, f)
    \pi_{kj} = v_{kj}*\prod_{l=1}^{k-1}(1-v_{kj})
    v_{kj} ~ beta(\alpha_0 \beta_k, alpha_0*(1-\sum_{l=1}^k \beta_l) )
    \alpha_0 ~ Ga(g, h)
    \mu_k ~ N(0, m\Sigma_k)
    \Sigma_j ~ IW(nu0+2, nu0*Phi_k)
    """
    def __init__(self, data, ncomp=256, gamma0=10, m0=None,
                 nu0=None, Phi0=None, e0=5, f0=0.1, g0=0.1, h0=0.1,
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 parallel=False, verbose=False):

        # regardless of data class or _has_gpu, initialize gpu data to None
        # this gets set in sample method if a gpu device is available
        self.gpu_data = None

        if not issubclass(type(data), HDPNormalMixture):
            self.parallel = parallel

            # get the data .. should add checks here later
            self.data = [np.asarray(d) for d in data]
            self.ngroups = len(self.data)
            self.ndim = self.data[0].shape[1]
            self.nobs = tuple([d.shape[0] for d in self.data])

            # need for ident code
            self.cumobs = np.zeros(self.ngroups+1, dtype=np.uint32)
            self.cumobs[1:] = np.asarray(self.nobs).cumsum()
            self.ncomp = ncomp

            if m0 is not None:
                if len(m0) == self.ndim:
                    self.mu_prior_mean = m0.copy()
                elif len(m0) == 1:
                    self.mu_prior_mean = m0*np.ones(self.ndim)
            else:
                self.mu_prior_mean = np.zeros(self.ndim)

                self.gamma = gamma0*np.ones(ncomp)

            self._set_initial_values(alpha0, nu0, Phi0, mu0, Sigma0,
                                     weights0, e0, f0)
            # initialize hdp specific vars
            if weights0 is None:
                self._weights0 = np.zeros(
                    (self.ngroups, self.ncomp),
                    dtype=np.float)
                self._weights0.fill(1/self.ncomp)
            else:
                self._weights0 = weights0.copy()
            self._stick_beta0 = stats.beta.rvs(
                1,
                self._alpha0,
                size=self.ncomp-1)
            self._beta0 = break_sticks(self._stick_beta0)
            self._alpha00 = 1.0
            self.e0, self.f0 = g0, h0
            # start out small? more accepts?
            self.prop_scale = 0.01 * np.ones(self.ncomp)
            self.prop_scale[-1] = 1.

        else:
            # get all important vars from input class
            self.data = data.data
            self.ngroups = data.ngroups
            self.nobs = data.nobs
            self.ndim = data.ndim
            self.ncomp = data.ncomp
            self.cumobs = data.cumobs.copy()
            self._weights0 = data.weights[-1].copy()
            self._stick_beta0 = data.stick_beta.copy()
            self._beta0 = break_sticks(self._stick_beta0)
            self.e0, self.f0 = data.e0, data.f0
            self.e, self.f = data.e, data.f
            self._nu0 = data._nu0
            self._Phi0 = data._Phi0
            self.mu_prior_mean = data.mu_prior_mean.copy()
            self.gamma = data.gamma.copy()
            self._alpha0 = data.alpha[-1].copy()
            self._alpha00 = data.alpha0[-1].copy()
            self._weights0 = data.weights[-1].copy()
            self._mu0 = data.mu[-1].copy()
            self._Sigma0 = data.Sigma[-1].copy()
            self.prop_scale = data.prop_scale.copy()
            self.parallel = data.parallel

        self.AR = np.zeros(self.ncomp)
        self.verbose = verbose

        # data working var
        self.alldata = np.empty((sum(self.nobs), self.ndim), dtype=np.double)
        for i in xrange(self.ngroups):
            self.alldata[self.cumobs[i]:self.cumobs[i+1], :] = \
                self.data[i].copy()

    def sample(
            self,
            niter=1000,
            nburn=100,
            thin=1,
            tune_interval=100,
            ident=False,
            device=None,
            callback=None
    ):
        """
        Performs MCMC sampling of the posterior. beta must be sampled
        using Metropolis Hastings and its proposal distribution will
        be tuned every tune_interval iterations during the burn-in
        period. It is suggested that an ample burn-in is used and the
        AR parameters stores the acceptance rate for the stick weights
        of beta and alpha_0.

        Parameters
        ----------
        niter
        nburn
        thin
        tune_interval
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

        self._setup_storage(niter, thin)

        alpha = self._alpha0
        alpha0 = self._alpha00
        weights = self._weights0
        beta = self._beta0
        stick_beta = self._stick_beta0
        mu = self._mu0
        sigma = self._Sigma0

        for i in range(-nburn, niter):
            if isinstance(self.verbose, int) and \
                    self.verbose and \
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

            # Get initial reference if needed
            if i == 0 and ident:
                z_ref = []
                for ii in xrange(self.ngroups):
                    z_ref.append(z_hat[ii].copy())
                c0 = np.zeros((self.ncomp, self.ncomp), dtype=np.double)
                for j in xrange(self.ncomp):
                    for ii in xrange(self.ngroups):
                        # noinspection PyTypeChecker
                        c0[j, :] += np.sum(z_ref[ii] == j)

            # update mu and sigma
            counts = self._update_mu_sigma(mu, sigma, labels)

            # update weights, masks
            stick_weights, weights = self._update_stick_weights(
                counts,
                beta,
                alpha0
            )

            stick_beta, beta = sampler.sample_beta(
                stick_beta,
                beta,
                stick_weights,
                alpha0,
                alpha,
                self.AR,
                self.prop_scale,
                self.parallel
            )

            # hyper-parameters
            alpha = self._update_alpha(stick_beta)
            alpha0 = sampler.sample_alpha0(
                stick_weights,
                beta,
                alpha0,
                self.e0,
                self.f0,
                self.prop_scale,
                self.AR
            )

            # Relabel
            if i > 0 and ident:
                cost = c0.copy()
                for Z, Zr in zip(z_hat, z_ref):
                    get_cost(Zr, Z, cost)
                _, iii = np.where(munkres(cost))
                beta = beta[iii]
                weights = weights[:, iii]
                mu = mu[iii]
                sigma = sigma[iii]

            # save
            if i >= 0:
                self.beta[i] = beta
                self.weights[i] = weights
                self.alpha[i] = alpha
                self.alpha0[i] = alpha0
                self.mu[i] = mu
                self.Sigma[i] = sigma
            elif (nburn + i + 1) % tune_interval == 0:
                self._tune(tune_interval)

    def _setup_storage(self, niter=1000, thin=1):
        n_results = niter // thin
        self.weights = np.zeros((n_results, self.ngroups, self.ncomp))
        self.beta = np.zeros((n_results, self.ncomp))
        self.mu = np.zeros((n_results, self.ncomp, self.ndim))
        self.Sigma = np.zeros((n_results, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.zeros(n_results)
        self.alpha0 = np.zeros(n_results)

    def _update_labels(self, mu, sigma, weights, ident=False):
        # gets the latent classifications
        z_hat = []
        if self.gpu_data is not None:
            return get_hdp_labels_GPU(
                self.gpu_data,
                weights,
                mu,
                sigma,
                ident
            )
        else:
            labels = [np.zeros(self.nobs[j]) for j in range(self.ngroups)]
            for j in xrange(self.ngroups):
                densities = mvn_weighted_logged(
                    self.data[j],
                    mu,
                    sigma,
                    weights[j]
                )
                labels[j] = sample_discrete(densities).squeeze()
                if ident:
                    z_hat.append(np.asarray(densities.argmax(1), dtype='i'))

        return labels, z_hat

    def _update_stick_weights(self, counts, beta, alpha0):
        new_weights = np.zeros((self.ngroups, self.ncomp))
        new_stick_weights = np.zeros((self.ngroups, self.ncomp-1))
        for j in xrange(self.ngroups):
            reverse_cum_sum = counts[j][::-1].cumsum()[::-1]

            a = alpha0 * beta[:-1] + counts[j][:-1]
            b = alpha0 * (1 - beta[:-1].cumsum()) + reverse_cum_sum[1:]
            sticks_j, weights_j = stick_break_proc(a, b)
            new_weights[j] = weights_j
            new_stick_weights[j] = sticks_j
        return new_stick_weights, new_weights

    # function to get log_post for beta
    @staticmethod
    def beta_post(stick_beta, beta, stick_weights, alpha0, alpha):
        log_post = 0

        a, b = alpha0 * beta[:-1], alpha0 * (1 - beta[:-1].cumsum())
        log_post += np.sum(stats.beta.logpdf(stick_weights, a, b))

        log_post += np.sum(stats.beta.logpdf(stick_beta, 1, alpha))
        return log_post

    def _update_beta(self, stick_beta, beta, stick_weights, alpha0, alpha):

        old_stick_beta = stick_beta.copy()
        for k in xrange(self.ncomp-1):
            # get initial log post
            log_post = self.beta_post(
                stick_beta,
                beta,
                stick_weights,
                float(alpha0),
                float(alpha)
            )

            # sample new beta from reflected normal
            prop = stats.norm.rvs(stick_beta[k], self.prop_scale[k])
            while prop > (1-1e-9) or prop < 1e-9:
                if prop > 1-1e-9:
                    prop = 2*(1-1e-9) - prop
                else:
                    prop = 2*1e-9 - prop
            stick_beta[k] = prop
            beta = break_sticks(stick_beta)

            # get new posterior
            log_post_new = self.beta_post(
                stick_beta,
                beta,
                stick_weights,
                float(alpha0),
                float(alpha)
            )

            # accept or reject
            if stats.expon.rvs() > log_post - log_post_new:
                # accept
                self.AR[k] += 1
            else:
                stick_beta[k] = old_stick_beta[k]
                beta = break_sticks(stick_beta)
        return stick_beta, beta

    def _update_mu_sigma(self, mu, sigma, labels, other_dat=None):
        all_labels = np.empty(self.cumobs[-1], dtype=np.int)
        i = 0
        for labs in labels:
            all_labels[self.cumobs[i]:self.cumobs[i + 1]] = labs.copy()
            i += 1
        if len(mu.shape) == 1:
            import pdb
            pdb.set_trace()

        # sample_mu_Sigma expects cumobs to be floats
        counts = sampler.sample_mu_Sigma(
            mu,
            sigma,
            all_labels,
            self.alldata,  # using combined all_data here
            self.gamma[0],
            self.mu_prior_mean,
            self._nu0,
            self._Phi0[0],
            self.parallel,
            self.cumobs[1:].astype(np.float64)  # convert to floats
        )

        return counts

    def _tune(self, tune_interval):
        """
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        for j in xrange(len(self.AR)):
            ratio = self.AR[j] / tune_interval
            if ratio < 0.001:
                self.prop_scale[j] *= np.sqrt(0.1)
            elif ratio < 0.05:
                self.prop_scale[j] *= np.sqrt(0.5)
            elif ratio < 0.2:
                self.prop_scale[j] *= np.sqrt(0.9)
            elif ratio > 0.95:
                self.prop_scale[j] *= np.sqrt(10)
            elif ratio > 0.75:
                self.prop_scale[j] *= np.sqrt(2)
            elif ratio > 0.5:
                self.prop_scale[j] *= np.sqrt(1.1)
            self.AR[j] = 0
