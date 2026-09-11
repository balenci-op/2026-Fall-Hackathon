"""Label-free PCA evidence from existing cached XYZ; no semantic classifier.

Covariance uses population normalization 1/N. Eigenvalues are descending.
Ratios follow CloudCompare CCCoreLib Neighbourhood::computeFeature. Principal
axis alignment uses v1; surface-normal alignment uses v3 and is a different
quantity. Eigenvector signs are arbitrary; absolute vertical dot products remove
that ambiguity. Repeated eigenvalues can make directions undefined.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree

PARAMETERS = {
    'radii': [0.3, 1.0], 'max_query_points_per_cloud': 512, 'query_seed': 20260911,
    'minimum_neighbors_including_self': 6, 'relative_eigengap_tolerance': 1e-6,
    'zero_variance_floor': 1e-20, 'covariance_normalization': 'population 1/N',
    'distance_units': 'unresolved source coordinate units',
    'descriptive_bins': {'linear_min': .6, 'planar_min': .6, 'scattered_min': .25,
                         'axis_vertical_min': .9},
    'note': 'Fixed before analysis; bins describe shape only, never semantic classes. Query fractions are point-weighted, not fractions of objects or area.'}

NAMES = ('linearity', 'planarity', 'scattering', 'axis_vertical_alignment',
         'normal_vertical_alignment', 'surface_verticality')


def pca_features(xyz):
    """Return explicit validity, covariance, descending eigenpairs and shape ratios."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('XYZ must be finite Nx3')
    result = {'count': len(xyz), 'status': 'insufficient_neighbors',
              'covariance': np.full((3, 3), np.nan), 'eigenvalues': np.full(3, np.nan),
              'eigenvectors_columns': np.full((3, 3), np.nan),
              'extents_xyz': np.ptp(xyz, axis=0) if len(xyz) else np.full(3, np.nan),
              **{name: float('nan') for name in NAMES}}
    if len(xyz) < PARAMETERS['minimum_neighbors_including_self']:
        return result
    shifted = xyz - xyz[0]  # Stable with large survey offsets.
    centered = shifted - shifted.mean(axis=0)
    covariance = centered.T @ centered / len(xyz)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    values = np.maximum(eigenvalues[::-1], 0)
    vectors = eigenvectors[:, ::-1].copy()
    for col in range(3):
        if vectors[np.argmax(np.abs(vectors[:, col])), col] < 0:
            vectors[:, col] *= -1
    result.update(covariance=covariance, eigenvalues=values, eigenvectors_columns=vectors)
    if values[0] <= PARAMETERS['zero_variance_floor']:
        result['status'] = 'zero_variance'
        return result
    l1, l2, l3 = values
    linear, planar, scatter = (l1-l2)/l1, (l2-l3)/l1, l3/l1
    gap = PARAMETERS['relative_eigengap_tolerance']
    axis = abs(vectors[2, 0]) if linear > gap else np.nan
    normal = abs(vectors[2, 2]) if planar > gap else np.nan
    result.update(status='valid', linearity=float(linear), planarity=float(planar),
                  scattering=float(scatter), axis_vertical_alignment=float(axis),
                  normal_vertical_alignment=float(normal), surface_verticality=float(1-normal))
    return result


def local_features(xyz, raw_rows):
    """Uniform reproducible query subset, both radius searches in the full role cloud.

    Candidate and context must call this separately: candidate neighborhoods never
    borrow context points. A query is included in its own radius neighborhood.
    No cap is applied to the returned neighbors; the query budget bounds work.
    """
    rng = np.random.default_rng(PARAMETERS['query_seed'])
    chosen = np.sort(rng.choice(len(xyz), min(len(xyz), PARAMETERS['max_query_points_per_cloud']), replace=False))
    tree = cKDTree(xyz)
    outputs = {}
    for radius in PARAMETERS['radii']:
        neighborhoods = tree.query_ball_point(xyz[chosen], radius, workers=1)
        results = [pca_features(xyz[neighbors]) for neighbors in neighborhoods]
        outputs[radius] = {'query_indices': chosen, 'query_raw_record_indices': np.asarray(raw_rows)[chosen],
            'query_xyz': xyz[chosen], 'neighbor_count': np.array([r['count'] for r in results]),
            'status': np.array([r['status'] for r in results], dtype='U24'),
            **{name: np.array([r[name] for r in results]) for name in
               (*NAMES, 'covariance', 'eigenvalues', 'eigenvectors_columns', 'extents_xyz')}}
    return outputs


def summarize_local(features):
    valid = features['status'] == 'valid'
    count = int(valid.sum())
    summary = {'queries': len(valid), 'valid': count,
               'insufficient_neighbors': int(np.sum(features['status']=='insufficient_neighbors')),
               'zero_variance': int(np.sum(features['status']=='zero_variance')),
               'neighbor_count_quantiles': np.quantile(features['neighbor_count'], [.1, .5, .9]).tolist()}
    for name in NAMES:
        values = features[name][valid & np.isfinite(features[name])]
        summary[name] = {'defined':len(values), 'q10_q50_q90':np.quantile(values,[.1,.5,.9]).tolist() if len(values) else [None]*3}
    bins = PARAMETERS['descriptive_bins']
    masks = {'linear': features['linearity']>=bins['linear_min'],
             'planar': features['planarity']>=bins['planar_min'],
             'scattered': features['scattering']>=bins['scattered_min'],
             'vertical_linear': (features['linearity']>=bins['linear_min']) &
                                (features['axis_vertical_alignment']>=bins['axis_vertical_min'])}
    summary['proportions_of_valid_queries'] = {name:float(np.sum(mask & valid)/count) if count else None for name,mask in masks.items()}
    return summary


def plain(value):
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')


def synthetic_validation():
    """Independent known shapes; no synthetic records enter real experiment outputs."""
    axis=np.linspace(-1,1,9)
    line=np.column_stack([np.zeros(9),np.zeros(9),axis])
    x,y=np.meshgrid(axis,axis);plane=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    cube=np.array(np.meshgrid([-1.,0,1.],[-1.,0,1.],[-1.,0,1.])).reshape(3,-1).T
    expected=[(line,[1,0,0]),(plane,[0,1,0]),(cube,[0,0,1])]
    for points,ratios in expected:
        result=pca_features(points)
        np.testing.assert_allclose([result[n] for n in NAMES[:3]],ratios,atol=1e-12)
        singular=np.linalg.svd(points-points.mean(axis=0),compute_uv=False)
        np.testing.assert_allclose(result['eigenvalues'],singular**2/len(points),atol=1e-12)
        shifted=pca_features(points+np.array([2454000.,414800.,800.]))
        np.testing.assert_allclose(result['eigenvalues'],shifted['eigenvalues'],atol=1e-10)
    assert pca_features(line)['axis_vertical_alignment']==1
    assert np.isnan(pca_features(line)['normal_vertical_alignment'])
    assert pca_features(plane)['normal_vertical_alignment']==1
    assert np.isnan(pca_features(plane)['axis_vertical_alignment'])
    assert np.isnan(pca_features(cube)['axis_vertical_alignment'])
    vertical_plane=plane[:,[0,2,1]]
    assert pca_features(vertical_plane)['normal_vertical_alignment']==0
    assert pca_features(np.zeros((6,3)))['status']=='zero_variance'
    assert pca_features(line[:2])['status']=='insufficient_neighbors'
    # Verify the local path uses all neighbors, not just its sampled query set.
    dense_line=np.column_stack([np.zeros(2000),np.zeros(2000),np.linspace(-1,1,2000)])
    local=local_features(dense_line,np.arange(2000))
    repeated=local_features(dense_line,np.arange(2000))
    for radius,features in local.items():
        np.testing.assert_array_equal(features['query_indices'],repeated[radius]['query_indices'])
        np.testing.assert_allclose(features['linearity'],1)
        assert features['neighbor_count'].max()>len(features['query_indices'])
    return {'status':'passed','separate_from_real_candidates':True,
            'checks':['line','horizontal plane','vertical plane','isotropic volume','SVD eigenvalue crosscheck',
                      'large coordinate translation','undefined tied-eigenvalue directions','coincident points',
                      'insufficient neighbors','reproducible local queries with full-cloud neighbor search']}
