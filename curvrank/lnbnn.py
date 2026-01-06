import time
import numpy as np
import annoy
from tqdm import tqdm
from typing import Literal


def build_lnbnn_index(
    data: np.ndarray, 
    fpath: str, 
    num_trees: int = 10
) -> annoy.AnnoyIndex:
    """
    Build an Annoy index from descriptor data and save it to disk.

    Args:
        data: Shape (n, d) array of n descriptors with dimension d.
        fpath: File path to save the index.
        num_trees: Number of trees for the index (more trees = better accuracy, slower build).

    Returns:
        The built Annoy index.
    """
    print(f"Building LNBNN index with {data.shape[0]} descriptors...")
    fdim = data.shape[1]
    index = annoy.AnnoyIndex(fdim, metric="euclidean")
    
    for i in tqdm(range(len(data)), desc="Adding items"):
        index.add_item(i, data[i])
    
    start = time.time()
    index.build(num_trees)
    elapsed = time.time() - start
    print(f"Index built in {elapsed:.2f}s")
    
    index.save(fpath)
    print(f"Index saved to {fpath}")
    
    return index


def lnbnn_identify(
    index_fpath: str,
    k: int,
    descriptors: np.ndarray,
    names: np.ndarray,
    search_k: int = -1
) -> dict[str, float]:
    """
    Perform LNBNN identification using a pre-built index.

    Implementation based on: www.cs.ubc.ca/~lowe/papers/12mccannCVPR.pdf

    Args:
        index_fpath: Path to the saved Annoy index file.
        k: Number of nearest neighbors to retrieve.
        descriptors: Shape (n, d) array of query descriptors.
        names: Array of names corresponding to each item in the index.
        search_k: Search parameter (-1 for default, higher = more accurate but slower).

    Returns:
        Dictionary mapping names to scores. More negative scores indicate stronger matches.
    """
    fdim = descriptors.shape[1]
    index = annoy.AnnoyIndex(fdim, metric="euclidean")
    index.load(index_fpath)

    unique_names = np.unique(names)
    scores = {name: 0.0 for name in unique_names}
    
    for descriptor in descriptors:
        ind, dist = index.get_nns_by_vector(
            descriptor, k + 1, search_k=search_k, include_distances=True
        )
        classes = np.array([names[idx] for idx in ind[:-1]])
        
        for c in np.unique(classes):
            (j,) = np.where(classes == c)
            score = dist[j.min()] - dist[-1]
            scores[c] += score

    return scores


def aggregate_descriptors_by_scale(
    descriptor_dicts: list[dict[float, np.ndarray]],
    names: list[str],
    scales: np.ndarray
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Aggregate descriptors from multiple images into per-scale arrays with labels.

    Args:
        descriptor_dicts: List of descriptor dictionaries (one per image).
        names: List of individual names corresponding to each descriptor dict.
        scales: Array of scales used for curvature computation.

    Returns:
        Dictionary mapping scale -> (descriptors array, names array).
    """
    lnbnn_data = {}
    
    for scale in scales:
        all_descriptors = []
        all_names = []
        
        for desc_dict, name in zip(descriptor_dicts, names):
            if desc_dict is not None and scale in desc_dict:
                descriptors = desc_dict[scale]
                if descriptors.shape[0] > 0:
                    all_descriptors.append(descriptors)
                    all_names.extend([name] * descriptors.shape[0])
        
        if all_descriptors:
            D = np.vstack(all_descriptors)
            N = np.array(all_names)
            lnbnn_data[scale] = (D, N)
        else:
            lnbnn_data[scale] = (np.empty((0, 32), dtype=np.float32), np.array([]))
    
    return lnbnn_data


def build_indices_for_view(
    lnbnn_data: dict[float, tuple[np.ndarray, np.ndarray]],
    view: Literal["left", "right"],
    output_dir: str
) -> dict[float, str]:
    """
    Build and save LNBNN indices for all scales for a specific view.

    Args:
        lnbnn_data: Dictionary mapping scale -> (descriptors, names).
        view: The ear view ("left" or "right").
        output_dir: Directory to save index files.

    Returns:
        Dictionary mapping scale -> index file path.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    index_paths = {}
    for scale, (descriptors, names) in lnbnn_data.items():
        if descriptors.shape[0] > 0:
            fpath = os.path.join(output_dir, f"{view}_{scale:.3f}.ann")
            build_lnbnn_index(descriptors, fpath)
            index_paths[scale] = fpath
        else:
            print(f"No descriptors for scale {scale}, skipping index build")
    
    return index_paths


def identify_query(
    query_descriptors: dict[float, np.ndarray],
    index_paths: dict[float, str],
    db_lnbnn_data: dict[float, tuple[np.ndarray, np.ndarray]],
    k: int = 2
) -> dict[str, float]:
    """
    Identify a query image against built indices.

    Args:
        query_descriptors: Dictionary mapping scale -> query descriptors.
        index_paths: Dictionary mapping scale -> index file path.
        db_lnbnn_data: Dictionary mapping scale -> (db descriptors, db names).
        k: Number of neighbors for LNBNN (paper recommends k=2).

    Returns:
        Aggregated scores across all scales. More negative = stronger match.
    """
    agg_scores: dict[str, float] = {}
    
    for scale, descriptors in query_descriptors.items():
        if scale not in index_paths or descriptors.shape[0] == 0:
            continue
            
        _, names = db_lnbnn_data[scale]
        if len(names) == 0:
            continue
            
        scores = lnbnn_identify(index_paths[scale], k, descriptors, names)
        
        for name, score in scores.items():
            if name not in agg_scores:
                agg_scores[name] = 0.0
            agg_scores[name] += score
    
    return agg_scores


def rank_results(scores: dict[str, float]) -> list[tuple[str, float]]:
    """
    Rank identification results by score.

    Args:
        scores: Dictionary mapping names to scores.

    Returns:
        List of (name, score) tuples sorted by score (most negative first).
    """
    return sorted(scores.items(), key=lambda x: x[1])

