"""
Written by: Andrew Cron
"""

import numpy as np
import gpustats
import gpustats.util as gpu_util
import gpustats.sampler as gpu_sampler
from pycuda.gpuarray import to_gpu
import cuda_functions


def init_GPUWorkers(data, device_number):
    """
    Send data to GPU device
    """

    gpu_util.threadSafeInit(device_number)
    gpu_data = []

    # dpmix and BEM
    if type(data) == np.ndarray:
        gpu_data.append(to_gpu(np.asarray(data, dtype=np.float32)))
    else:  # HDP...one or more data sets per GPU
        for i in xrange(len(data)):
            gpu_data.append(to_gpu(np.asarray(data[i], dtype=np.float32)))

    return gpu_data


def get_hdp_labels_GPU(gpu_data, w, mu, Sigma, relabel=False):

    labels = []
    Z = []

    for i, data_set in enumerate(gpu_data):
        densities = gpustats.mvnpdf_multi(
            data_set,
            mu,
            Sigma,
            weights=w[i].flatten(),
            get=False,
            logged=True,
            order='C'
        )

        if relabel:
            Z.append(
                np.asarray(
                    cuda_functions.gpu_apply_row_max(densities)[1].get(),
                    dtype='i'
                )
            )
        else:
            Z.append(None)

        labels.append(
            np.asarray(
                gpu_sampler.sample_discrete(densities, logged=True),
                dtype='i'
            )
        )

        densities.gpudata.free()
        del densities

    return labels, Z


def get_labelsGPU(gpu_data, w, mu, Sigma, relabel=False):
    densities = gpustats.mvnpdf_multi(
        gpu_data[0],
        mu,
        Sigma,
        weights=w.flatten(),
        get=False,
        logged=True,
        order='C'
    )

    if relabel:
        Z = np.asarray(
            cuda_functions.gpu_apply_row_max(densities)[1].get(),
            dtype='i'
        )
    else:
        Z = None

    labels = np.asarray(
        gpu_sampler.sample_discrete(densities, logged=True),
        dtype='i'
    )

    densities.gpudata.free()
    del densities

    return labels, Z


def get_expected_labels_GPU(gpu_data, w, mu, Sigma):
    densities = gpustats.mvnpdf_multi(
        gpu_data[0],
        mu,
        Sigma,
        weights=w.flatten(),
        get=False,
        logged=True,
        order='C'
    )

    dens = np.asarray(densities.get(), dtype='d')

    densities.gpudata.free()

    return dens
