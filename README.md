# Controlled k-means

Controlled k-means is a variant of k-means algorithm for the minimum sum-of-squares clustering (MSSC). In this variant, the lassical centroid update rule of standard k-means is revisited.

As Controlled k-means algorithm is designed like standard k-means, the source code of the scikit-learn k-means implementation structure is used. https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/cluster/_kmeans.py#L1192.


# Run
To run the algorithm, the example file shows an exemple.

# Parameters of the algorithm

## Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `n_clusters` | int | - | Number of clusters |
| `init` | str or ndarray | `"k-means++"` | Initialization method (`"k-means++"`, `"random"` or custom array of centers) |
| `n_init` | int | 10 | Number of independent runs |
| `init_random_seed` | int or None | None | Seed for initialization randomness |
| `max_iter` | int or None | None | Maximum iterations per run |
| `verbose` | bool | False | Show progress messages |
| `tol` | float | 1e-4 | Convergence tolerance |
| `algorithm` | str | `"controlled"` | `"controlled"` or `"hybrid-controlled"` |
| `alpha_interval` | tuple(float, float) | (1.0, 1.9) | Interval for random α values |
| `alpha_random_seed` | int or None | None | Seed for alpha randomness |

