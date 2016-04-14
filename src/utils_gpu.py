"""
Written by: Andrew Cron
"""

import numpy as np

# noinspection PyPackageRequirements
import gpustats

# noinspection PyPackageRequirements
import gpustats.util as gpu_util

# noinspection PyPackageRequirements
import gpustats.sampler as gpu_sampler

# noinspection PyPackageRequirements
from pycuda.gpuarray import to_gpu
import cuda_functions


def init_gpu_data(data, device_number):
    """
    Send data to GPU device
    """

    gpu_util.threadSafeInit(device_number)
    gpu_data = []

    # DP and BEM
    if type(data) == np.ndarray:
        gpu_data.append(to_gpu(np.asarray(data, dtype=np.float32)))
    else:  # HDP...one or more data sets per GPU
        for i in xrange(len(data)):
            gpu_data.append(to_gpu(np.asarray(data[i], dtype=np.float32)))

    return gpu_data


def get_hdp_labels_gpu(gpu_data, w, mu, sigma, relabel=False):

    labels = []
    z = []

    for i, data_set in enumerate(gpu_data):
        densities = gpustats.mvnpdf_multi(
            data_set,
            mu,
            sigma,
            weights=w[i].flatten(),
            get=False,
            logged=True,
            order='C'
        )

        if relabel:
            z.append(
                np.asarray(
                    cuda_functions.gpu_apply_row_max(densities)[1].get(),
                    dtype='i'
                )
            )
        else:
            z.append(None)

        labels.append(
            np.asarray(
                gpu_sampler.sample_discrete(densities, logged=True),
                dtype='i'
            )
        )

        densities.gpudata.free()
        del densities

    return labels, z


def get_dp_labels_gpu(gpu_data, w, mu, sigma, relabel=False):
    densities = gpustats.mvnpdf_multi(
        gpu_data[0],
        mu,
        sigma,
        weights=w.flatten(),
        get=False,
        logged=True,
        order='C'
    )

    if relabel:
        z = np.asarray(
            cuda_functions.gpu_apply_row_max(densities)[1].get(),
            dtype='i'
        )
    else:
        z = None

    labels = np.asarray(
        gpu_sampler.sample_discrete(densities, logged=True),
        dtype='i'
    )

    densities.gpudata.free()
    del densities

    return labels, z


def get_bem_densities_gpu(gpu_data, w, mu, sigma):
    densities = gpustats.mvnpdf_multi(
        gpu_data[0],
        mu,
        sigma,
        weights=w.flatten(),
        get=False,
        logged=True,
        order='C'
    )

    dens = np.asarray(densities.get(), dtype='d')

    densities.gpudata.free()

    return dens
