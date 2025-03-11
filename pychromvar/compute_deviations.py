from anndata import AnnData
from mudata import MuData
from typing import Union
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import logging

def process_background_set(args):
    """Compute Y' for one of the background sets of peaks"""
    i, shm_name, X_shape, fraction, per_cell_sum, bg_peaks, M_shape, M_data, M_indices, M_indptr = args

    # Load shared memory for X
    shm = shared_memory.SharedMemory(name=shm_name)
    X = np.ndarray(X_shape, dtype=np.float32, buffer=shm.buf)

    # Reconstruct sparse matrix M
    M = csr_matrix((M_data, M_indices, M_indptr), shape=M_shape)

    # Replace peaks for this background set
    replacement_idx = bg_peaks[:, i]
    rows, cols = M.nonzero()
    new_rows = replacement_idx[rows]
    M_r = csr_matrix((M.data, (new_rows, cols)), shape=M.shape)

    # Compute (MxB)xE' 
    obs_fragments = X @ M_r
    tf_fraction = M_r.multiply(fraction[:, np.newaxis]).sum(axis=0).A1
    e_fragments = (per_cell_sum[:, np.newaxis] * tf_fraction)
    
    # Compute MxE' 
    tf_fraction_ = M.multiply(fraction[:, np.newaxis]).sum(axis=0).A1
    e_fragments_ = (per_cell_sum[:, np.newaxis] * tf_fraction_)
    
    # Conmpute Y'
    Y_r = (obs_fragments - e_fragments) / e_fragments_

    return Y_r

def compute_Y(X, M, fraction, n_fragments):
    """Compute Y matrix"""
    obs_fragments = X @ M
    tf_fraction = M.multiply(fraction[:, np.newaxis]).sum(axis=0).A1
    e_fragments = (n_fragments[:, np.newaxis] * tf_fraction)
    Y = (obs_fragments - e_fragments) / e_fragments

    return Y

def process_background_set_single(X, M, fraction, per_cell_sum, bg_peaks, i):
    """Function for computing Y' for a single background set without shared memory"""

    # Replace peaks for this background set
    replacement_idx = bg_peaks[:, i]
    rows, cols = M.nonzero()
    new_rows = replacement_idx[rows]
    M_r = csr_matrix((M.data, (new_rows, cols)), shape=M.shape)

    # Compute (MxB)xE' 
    obs_fragments = X @ M_r
    tf_fraction = M_r.multiply(fraction[:, np.newaxis]).sum(axis=0).A1
    e_fragments = (per_cell_sum[:, np.newaxis] * tf_fraction)
    
    # Compute MxE' 
    tf_fraction_ = M.multiply(fraction[:, np.newaxis]).sum(axis=0).A1
    e_fragments_ = (per_cell_sum[:, np.newaxis] * tf_fraction_)
    
    # Conmpute Y'
    Y_r = (obs_fragments - e_fragments) / e_fragments_

    return Y_r

def compute_deviations(data: Union[AnnData, MuData], n_jobs=-1) -> AnnData:

    """Compute bias-corrected deviations.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    n_jobs : int, optional
        Number of cpus used for motif matching. If set to -1, all cpus will be used. Default: -1.
        
    Returns
    -------
    Anndata
        An anndata object containing estimated deviations.
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError("Expected AnnData or MuData object with 'atac' modality")
    
    assert "bg_peaks" in adata.varm_keys(), "Cannot find background peaks in the input object."
    assert "motif_match" in adata.varm_keys(), "Cannot find motif_match in the input object."

    # X should be dense and M should be csr
    if not isinstance(adata.X, csr_matrix):
        X = adata.X
    else:
        X = adata.X.toarray()

    if not isinstance(adata.varm['motif_match'], csr_matrix):
        M = csr_matrix(adata.varm['motif_match'])
    else:
        M = adata.varm['motif_match']

    bg_peaks = adata.varm['bg_peaks']
    N_bkg_sets = bg_peaks.shape[1]

    # Make the vectors to reconstruct E 
    per_peak_sum = np.array(X.sum(axis=0)).flatten()
    total = np.sum(per_peak_sum)
    fraction = per_peak_sum / total

    per_cell_sum = np.array(X.sum(axis=1)).flatten()

    logging.info('computing raw deviations...')

    # Compute Y
    Y = compute_Y(X, M, fraction, per_cell_sum)

    if n_jobs == 1:

        logging.info('n_jobs = 1. Not using multiprocessing...')
        results = []
        for i in tqdm(range(N_bkg_sets)):
            results.append(process_background_set_single(X, M, fraction, per_cell_sum, bg_peaks, i))
        
        mean_Y = np.mean(results, axis=0)
        std_Y = np.std(results, axis=0)

    else:

        logging.info('launching parallel jobs to compute background devs...')
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Copy X to shared memory
        shm_X = shared_memory.SharedMemory(create=True, size=X.nbytes)
        shared_array = np.ndarray(X.shape, dtype=X.dtype, buffer=shm_X.buf)
        np.copyto(shared_array, X)  

        # Prepare arguments for multiprocessing
        args = [
            (
                i,
                shm_X.name,
                X.shape,
                fraction,
                per_cell_sum,
                bg_peaks,
                M.shape,
                M.data,
                M.indices,
                M.indptr,
            )
            for i in range(N_bkg_sets)
        ]

        # Parallel computation
        with Pool(n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(process_background_set, args),
                    total=N_bkg_sets,
                    desc="processing background peak sets",
                )
            )

        # Compute mean and standard deviation
        logging.info('computing mean and std dev of background devs...')
        mean_Y = np.mean(results, axis=0)
        std_Y = np.std(results, axis=0)

        # Cleanup shared memory
        shm_X.close()
        shm_X.unlink()

    # Compute bias-corrected deviations
    logging.info('calculate bias corrected deviations...')
    dev = (Y - mean_Y) / std_Y
    dev = np.nan_to_num(dev, 0)

    # Create AnnData
    dev = AnnData(dev)
    dev.obs_names = adata.obs_names
    dev.var_names = adata.uns['motif_name']

    return dev
