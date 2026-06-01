# Algorithm selection for aero-engine flight mission profile clustering

Date: 2026-06-01

## Conclusion

Do not only cluster raw time series. For aero-engine mission profile and load-spectrum analysis, a more stable paper route is a dual-baseline framework:

1. Physical / damage feature clustering.
2. Multivariate time-series clustering using DTW / Soft-DTW.
3. HDBSCAN / OPTICS for abnormal mission detection.

## Candidate algorithms and libraries

| Candidate algorithm / library | Multivariate time-series suitability | DTW / Soft-DTW | Need preset cluster number | Noise / outliers | Center / representative sample | Damage weight / physical constraint | Maturity / baseline / improvement space | Recommendation |
|---|---|---|---|---|---|---|---|---|
| tslearn TimeSeriesKMeans | Strong; supports input shape `(n_ts, sz, d)` | Supports DTW and Soft-DTW | Yes | Weak | DTW / Soft-DTW barycenter | Can be modified through variable scaling, weighted distance, stage constraints | Very suitable as main baseline; improvement space is damage-weighted DTW | A |
| aeon TimeSeriesKMedoids / KASBA | Strong; designed for time-series clustering | Supports elastic distance / barycenter | Yes | Medium; medoid is more robust than mean | Medoid or elastic barycenter | Suitable for custom distance and representative mission selection | New but systematic; suitable for cross-validation with tslearn | A |
| Feature engineering + KMeans / MiniBatchKMeans | Strong after task-level feature extraction | No | Yes | Weak | Feature centroid + nearest real mission | Easiest route to include Miner damage, rainflow count, exceedance duration, load weights | Required traditional baseline; scikit-learn supports `sample_weight` in relevant estimators | A |
| Feature engineering + GMM | Strong; suitable for probabilistic mission family membership | No | Yes | Medium; low likelihood can indicate abnormal tasks | Mean, covariance, posterior probability | Usually implemented through feature design or resampling | Suitable as probabilistic baseline and soft-classification explanation | A- |
| HDBSCAN | Directly suitable for features; can also use precomputed distance | Can use precomputed DTW matrix | No | Strong; noise labeled as `-1` | Supports centroid / medoid-style representative analysis | Can include damage weights through custom distance matrix | Good for unknown cluster number and abnormal tasks; not ideal for smooth center spectrum | A- |
| DTAIDistance DTW + KMeans / KMedoids / hierarchical | Supports multidimensional DTW with `use_ndim` | Supports DTW / DBA | Usually yes | Medium | DBA or medoid | Custom DTW parameters are convenient | Focused DTW implementation; engineering interface weaker than tslearn / aeon | B+ |
| AgglomerativeClustering + precomputed DTW / damage distance | Strong depending on distance matrix | Supported externally | Cluster number or distance threshold | Weak | Dendrogram + medoid | Easy to use physical distance matrix | Good interpretability; O(n²) cost for large samples | B |
| OPTICS / DBSCAN | Strong in feature space; raw sequence requires distance matrix | Can use precomputed distance | No | Strong | No natural center; use medoid | Easy to connect custom distance | DBSCAN is sensitive to `eps`; OPTICS is better for multi-density exploration | B |
| SpectralClustering + DTW / Soft-DTW affinity | Can handle non-convex structures | Convert distance to affinity externally | Yes | Weak | No direct center; use representative sample | Damage distance can be converted to similarity | Suitable for small-sample comparison; interpretability and scalability are average | B- |
| tslearn / aeon KShape | Supports multivariate input but requires equal length | No; shape-based similarity | Yes | Weak | Shape centroid | Amplitude can be weakened by normalization | Suitable for shape-mode comparison, not for damage-dominated mission profiles | B- |
| TICC / fast-ticc | Suitable for multivariate state / stage segmentation | No | Yes or BIC | Weak | Sparse inverse-covariance network | Can include physical variables, but objective function modification is difficult | Suitable for flight-stage slicing, not first choice for complete mission-profile clustering | B |
| Time2Feat | Interpretable time-series feature extraction | No | Yes | Weak | Feature contribution explanation | Can add manual / physical feature selection | Good idea source, but code maturity is not ideal for main baseline | C |

## Reproduction order

1. Feature clustering baseline: duration, phase ratio, speed / temperature / altitude / Mach statistics, exceedance duration, rainflow cycles, Miner damage, equivalent load. Run KMeans, GMM, HDBSCAN.
2. Raw or phase-aligned sequence baseline: use `tslearn.clustering.TimeSeriesKMeans(metric="dtw")` and `metric="softdtw"`.
3. Representative mission profile extraction: output barycenter for each cluster and keep the nearest real flight mission as interpretable representative sample.
4. Abnormal mission processing: use HDBSCAN / OPTICS on damage-weighted features or weighted DTW distance matrix.

## Most promising paper improvement

Damage-weighted multivariate DTW / Soft-DTW + medoid / real representative mission constraint + physical stage constraint.

This is more relevant to load-spectrum compilation and accelerated mission test design than ordinary classification or ordinary KMeans.
