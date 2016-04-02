"""Find matching wave by simple correlation."""

import numpy as np
import os
import my_utils as mu


def __print_best_match(name, candidate, possibilities):
    number_of_samples = 20000
    max_value = 0
    max_name = "none"
    sig_noise = np.resize(candidate, number_of_samples)
    for k in possibilities:
        sig = np.resize(possibilities[k], number_of_samples)
        corr = np.corrcoef(sig, sig_noise)
        score = corr[0, 1]
        if score > max_value:
            max_name = k
            max_value = score
    print(os.path.basename(name) + "==" + os.path.basename(max_name))


def main():
    """Script starts here."""
    locations = mu.getlocationofdata()
    print(locations.__dict__)
    originscsvs = mu.readlistoffiles(locations.originfiles)
    filterdcsvs = mu.readlistoffiles(locations.filteredfiles)
    for k in originscsvs:
        __print_best_match(k, originscsvs[k], filterdcsvs)

if __name__ == "__main__":
    main()
