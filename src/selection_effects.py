import h5py as h5
import numpy as np
from pomegranate import GeneralMixtureModel as pomGMM
from pomegranate import MultivariateGaussianDistribution as pomMVG
import itertools as itt


class sel_effects():
    def __init__(self, pdetfname, paramnames):
        file = h5.File(pdetfname, 'r+')
        for name in paramnames:
            assert name in list(file.keys()), f"{name} is not in {pdetfname}"
        file.close()

        with h5.File(pdetfname, 'r+') as inp:
            self.m1s = np.array(inp['m1s'])
            self.m2s = np.array(inp['m2s'])
            self.pdet = np.array(inp['pdets'])
            # Check pdet normalization
            temp = np.trapz(y=self.pdet, x=self.m1s, axis=-1)
            norm = np.trapz(y=temp, x=self.m1s, axis=-1)
            assert np.isclose(norm, 1, atol=1e-04), "pdet is not normalized!"
            self.points = np.array(list(itt.product(self.m1s, self.m2s)))

    def ppop(self, **popparams):
        means = popparams['means']
        covs = popparams['covs']
        weights = popparams['weights']
        comps = popparams['components']
        model = pomGMM([pomMVG(means[i], covs[i]) for i in range(comps)],
                       weights=weights)
        return model.probability(self.points).reshape(self.pdet.shape)

    def integrate(self, **ppopparams):
        vals = np.exp(np.log(self.ppop(**ppopparams)) + np.log(self.pdet))
        temp = np.trapz(y=vals, x=self.m2s, axis=-1)  # Use trapz...
        integral = np.trapz(y=temp, x=self.m1s, axis=-1)  # ...twice
        return integral
