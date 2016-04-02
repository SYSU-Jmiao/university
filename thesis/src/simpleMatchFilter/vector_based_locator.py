"""Find matching wave by vectors distance."""

import my_utils as mu
import numpy as np
import gram_schmidt as gs
from scipy.spatial import distance


def __find_best_match(candidate, possiblities):
    min_distance = np.inf
    index = 0
    for idx, val in enumerate(possiblities.T):
        temp = distance.euclidean(candidate, val)
        if temp <= min_distance:
            min_distance = temp
            index = idx
    return index


def main():
    """Script starts here."""

    sampling_size = 20000
    locations = mu.getlocationofdata()
    print(locations.__dict__)
    originscsvs = mu.readlistoffiles(locations.originfiles)
    filterdcsvs = mu.readlistoffiles(locations.filteredfiles)
    m_origin = mu.create_matrix_from_filelist(originscsvs, sampling_size)
    m_filtered = mu.create_matrix_from_filelist(filterdcsvs, sampling_size)
    q_origin, r_origin = gs.gram_schmidt(m_origin)
    print(q_origin.shape, r_origin.shape)
    r_filtered = np.dot(q_origin.T, m_filtered)
    print("r_filtered:", r_filtered.shape)
    print("r_origin:", r_origin.shape)
    for colume in r_origin.T:
        answer = __find_best_match(colume, r_filtered)
        print("found:", answer)

if __name__ == "__main__":
    main()
