"""
Written by: Andrew Cron
"""

from mpi4py import MPI
import numpy as np
from utils import MCMC_Task
import gpustats
import gpustats.util as gpu_util
import gpustats.sampler as gpu_sampler
from pycuda.gpuarray import to_gpu
import cuda_functions


# Multi GPU
_datadevmap = {}
_dataind = {}


def init_GPUWorkers(data, device_number):
    """
    Send data to GPU device
    """

    # initialize GPU with its own random seed...not sure this is necessary now?
    # this allows for reproducible runs if the calling application
    # sets its own random seed
    random_seed = np.random.randint(2 ** 31)

    gpu_util.threadSafeInit(device_number)
    gpu_data = []

    # dpmix and BEM
    if type(data) == np.ndarray:
        gpu_data.append(to_gpu(np.asarray(data, dtype=np.float32)))
    else:  # HDP...one or more data sets per GPU
        for i in xrange(len(data)):
            gpu_data.append(to_gpu(np.asarray(data[i], dtype=np.float32)))

            _dataind[i] = i  # not sure if this is correct
            _datadevmap[i] = 0  # or this

    return gpu_data


def get_hdp_labels_GPU(workers, w, mu, Sigma, relabel=False):

    ndev = workers.remote_group.size
    ndata = len(_datadevmap)

    tasks = []
    for _i in xrange(ndev):
        tasks.append([])
    labels = []
    Z = []
    for _i in xrange(ndata):
        labels.append(None)
        Z.append(None)

    # setup task
    for i in xrange(ndata):
        tasks[_datadevmap[i]].append(
            MCMC_Task(Sigma.shape[0], relabel, _dataind[i], i)
        )

    for i in xrange(ndev):
        # send the number of tasks
        tsk = np.array(1, dtype='i')
        workers.Isend([tsk, MPI.INT], dest=i, tag=11)
        numtasks = np.array(len(tasks[i]), dtype='i')
        workers.Send([numtasks, MPI.INT], dest=i, tag=12)

        for tsk in tasks[i]:
            params = np.array(
                [
                    tsk.dataind,
                    tsk.ncomp,
                    int(tsk.relabel) + 1,
                    tsk.gid
                ],
                dtype='i'
            )
            workers.Send([params, MPI.INT], dest=i, tag=13)

            workers.Send(
                [np.asarray(w[tsk.gid].copy(), dtype='d'), MPI.DOUBLE],
                dest=i,
                tag=21
            )
            workers.Send(
                [np.asarray(mu, dtype='d'), MPI.DOUBLE],
                dest=i,
                tag=22
            )
            workers.Send(
                [np.asarray(Sigma, dtype='d'), MPI.DOUBLE],
                dest=i,
                tag=23
            )

    # wait for results from any device in any order ... 
    res_devs = [_i for _i in range(ndev)]
    while len(res_devs) > 0:
        for i in res_devs:
            if workers.Iprobe(source=i, tag=13):
                numres = np.array(0, dtype='i')
                workers.Recv([numres, MPI.INT], source=i, tag=13)

                for it in range(numres):
                    rnobs = np.array(0, dtype='i')
                    workers.Recv([rnobs, MPI.INT], source=i, tag=21)
                    labs = np.empty(rnobs, dtype='i')
                    workers.Recv([labs, MPI.INT], source=i, tag=22)
                    rgid = np.array(0, dtype='i')
                    workers.Recv([rgid, MPI.INT], source=i, tag=23)
                    labels[rgid] = labs
                    if relabel:
                        cZ = np.empty(rnobs, dtype='i')
                        workers.Recv([cZ, MPI.INT], source=i, tag=24)
                        Z[rgid] = cZ
                res_devs.remove(i)
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


def get_expected_labels_GPU(workers, w, mu, Sigma):
    # run all the threads
    ndev = workers.remote_group.size
    ncomp = len(w)
    ndim = Sigma.shape[1]
    nobs, i = 0, 0
    partitions = [0]

    for i in xrange(ndev):
        # give new params
        task = np.array(1, dtype='i')
        workers.Isend([task, MPI.INT], dest=i, tag=11)
        numtasks = np.array(1, dtype='i')
        workers.Send([numtasks, MPI.INT], dest=i, tag=12)
        params = np.array([0, len(w), 0], dtype='i')
        workers.Send([params, MPI.INT], dest=i, tag=13)

        # give bigger params
        workers.Send([np.asarray(w, dtype='d'), MPI.DOUBLE], dest=i, tag=21)
        workers.Send([np.asarray(mu, dtype='d'), MPI.DOUBLE], dest=i, tag=22)
        workers.Send([np.asarray(Sigma, dtype='d'), MPI.DOUBLE], dest=i, tag=23)

    # gather results
    xbars = []
    densities = []
    cts = []
    ll = 0

    for i in xrange(ndev):
        numres = np.array(0, dtype='i')
        workers.Recv(numres, source=i, tag=13)
        rnobs = np.array(0, dtype='i')
        workers.Recv(rnobs, source=i, tag=21)

        nobs += rnobs
        partitions.append(nobs)
        ct = np.empty(ncomp, dtype='d')
        workers.Recv([ct, MPI.DOUBLE], source=i, tag=22)
        cts.append(ct)
        xbar = np.empty(ncomp*ndim, dtype='d')
        workers.Recv([xbar, MPI.DOUBLE], source=i, tag=23)
        xbars.append(xbar.reshape(ncomp, ndim))
        dens = np.empty(rnobs*ncomp, dtype='d')
        workers.Recv([dens, MPI.DOUBLE], source=i, tag=24)
        densities.append(dens.reshape(rnobs, ncomp))
        nll = np.array(0, dtype='d')
        workers.Recv([nll, MPI.DOUBLE], source=i, tag=25)
        ll += nll
        gid = np.array(0, dtype='i')
        workers.Recv([gid, MPI.INT], source=i, tag=26)

    dens = np.zeros((nobs, ncomp), dtype='d')
    xbar = np.zeros((ncomp, ndim), dtype='d')
    ct = np.zeros(ncomp, dtype='d')

    for i in xrange(ndev):
        ct += cts[i]
        xbar += xbars[i]
        dens[partitions[i]:partitions[i+1], :] = densities[i]
        
    return ll, ct, xbar, dens


def kill_GPUWorkers(workers):
    # poison pill to each child
    ndev = workers.remote_group.size
    msg = np.array(-1, dtype='i')
    for i in xrange(ndev):
        workers.Isend([msg, MPI.INT], dest=i, tag=11)
    workers.Disconnect()
