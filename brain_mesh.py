from __future__ import annotations

import numpy as np

# Optional deps (we'll use them if available)
try:
    from nilearn import datasets, surface
    _HAS_NILEARN = True
except Exception:
    _HAS_NILEARN = False

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ============================================================
# FSAVERAGE CORTEX (realistic whole brain)
# ============================================================

def load_fsaverage_mesh(mesh: str = "fsaverage5") -> dict:
    """
    Returns mesh dict with coords/faces for left and right pial surfaces.

    If nilearn is unavailable, raises ImportError.
    """
    if not _HAS_NILEARN:
        raise ImportError("nilearn is required for load_fsaverage_mesh(). Install nilearn to use fsaverage.")

    fs = datasets.fetch_surf_fsaverage(mesh=mesh)
    coords_l, faces_l = surface.load_surf_mesh(fs.pial_left)
    coords_r, faces_r = surface.load_surf_mesh(fs.pial_right)
    return {
        "coords_l": np.asarray(coords_l, dtype=float),
        "faces_l": np.asarray(faces_l, dtype=int),
        "coords_r": np.asarray(coords_r, dtype=float),
        "faces_r": np.asarray(faces_r, dtype=int),
        "fs": fs,  # keep reference for optional atlas fetches
    }


# ============================================================
# PARCELLATION
# ============================================================

def make_parcels_from_coords(coords: np.ndarray, n_parcels: int = 120, seed: int = 7):
    """
    Coarse parcelization: k-means over vertex coordinates.
    Returns:
      labels: parcel_id per vertex in [1..n_parcels]
      centers: parcel centers
    """
    if not _HAS_SK:
        # fallback: deterministic angular binning (no sklearn dependency)
        labels = _parcellate_by_angles(coords, n_parcels=n_parcels)
        centers = _centers_from_labels(coords, labels, n_parcels=n_parcels)
        return labels.astype(int), centers.astype(float)

    km = MiniBatchKMeans(
        n_clusters=n_parcels,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
    )
    labels0 = km.fit_predict(coords).astype(int)  # 0..n-1
    centers = km.cluster_centers_
    # convert to 1..n
    labels = labels0 + 1
    return labels.astype(int), centers.astype(float)


def _parcellate_by_angles(coords: np.ndarray, n_parcels: int = 120) -> np.ndarray:
    """
    Deterministic fallback parcellation without sklearn.
    Returns parcel ids 1..n_parcels.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    th = np.arccos(np.clip(z / r, -1.0, 1.0))  # 0..pi
    ph = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)  # 0..2pi

    bins_lat = int(np.sqrt(n_parcels))
    bins_lon = int(np.ceil(n_parcels / bins_lat))

    bi = np.clip((th / np.pi * bins_lat).astype(int), 0, bins_lat - 1)
    bj = np.clip((ph / (2 * np.pi) * bins_lon).astype(int), 0, bins_lon - 1)

    pid = bi * bins_lon + bj + 1
    pid = np.minimum(pid, n_parcels)
    return pid.astype(int)


def _centers_from_labels(coords: np.ndarray, labels: np.ndarray, n_parcels: int) -> np.ndarray:
    centers = np.zeros((n_parcels, 3), dtype=float)
    for pid in range(1, n_parcels + 1):
        idx = labels == pid
        if np.any(idx):
            centers[pid - 1] = coords[idx].mean(axis=0)
        else:
            centers[pid - 1] = coords.mean(axis=0)
    return centers


# ============================================================
# LOBE LOCALIZATION (Destrieux → lobes)  ✅ solves “lobes not localized”
# ============================================================

def _destrieux_to_lobe(name: str) -> str:
    """
    Map Destrieux region names to coarse lobes.
    This is a heuristic mapping (good for demo + UI grouping).
    """
    n = name.lower()

    # Occipital
    if "occip" in n or "calcarine" in n or "cuneus" in n or "lingual" in n:
        return "OCCIPITAL"

    # Temporal
    if "temporal" in n or "fusiform" in n or "heschl" in n or "planum" in n:
        return "TEMPORAL"

    # Parietal
    if "pariet" in n or "supramarginal" in n or "angular" in n or "precuneus" in n:
        return "PARIETAL"

    # Frontal
    if "frontal" in n or "precentral" in n or "orbital" in n:
        return "FRONTAL"

    # Insula / cingulate / limbic-ish
    if "insula" in n:
        return "INSULA"
    if "cingulate" in n:
        return "CINGULATE"

    return "OTHER"


def get_lobe_labels_fsaverage(mesh: str = "fsaverage5") -> dict:
    """
    Returns:
      lobe_l: per-vertex lobe label (string) for left hemisphere
      lobe_r: per-vertex lobe label (string) for right hemisphere

    Uses Destrieux atlas if available. Falls back to heuristic by coordinates if atlas fetch fails.
    """
    if not _HAS_NILEARN:
        raise ImportError("nilearn is required for get_lobe_labels_fsaverage().")

    fs = datasets.fetch_surf_fsaverage(mesh=mesh)

    try:
        # Destrieux atlas on fsaverage: gives labels + LUT
        atlas = datasets.fetch_atlas_surf_destrieux()
        # atlas.maps_left / maps_right are file paths (gii)
        lab_l = surface.load_surf_data(atlas.maps_left).astype(int)
        lab_r = surface.load_surf_data(atlas.maps_right).astype(int)

        # atlas.labels: list of region names aligned with label indices
        labels = list(atlas.labels)

        # Build lobe name per label index
        lobe_by_idx = []
        for nm in labels:
            lobe_by_idx.append(_destrieux_to_lobe(str(nm)))

        # Convert vertex labels to lobe strings
        lobe_l = np.array([lobe_by_idx[i] if 0 <= i < len(lobe_by_idx) else "OTHER" for i in lab_l], dtype=object)
        lobe_r = np.array([lobe_by_idx[i] if 0 <= i < len(lobe_by_idx) else "OTHER" for i in lab_r], dtype=object)

        return {"lobe_l": lobe_l, "lobe_r": lobe_r, "atlas_name": "destrieux"}

    except Exception:
        # Fallback: coordinate-based coarse lobes (very rough)
        coords_l, _ = surface.load_surf_mesh(fs.pial_left)
        coords_r, _ = surface.load_surf_mesh(fs.pial_right)
        return {
            "lobe_l": _lobe_from_coords(np.asarray(coords_l, float), hemi="L"),
            "lobe_r": _lobe_from_coords(np.asarray(coords_r, float), hemi="R"),
            "atlas_name": "heuristic",
        }


def _lobe_from_coords(coords: np.ndarray, hemi: str) -> np.ndarray:
    """
    Very rough fallback: use y (anterior/posterior) and z (superior/inferior)
    for coarse lobe-like partitioning. This is only used if atlas fetch fails.
    """
    y = coords[:, 1]
    z = coords[:, 2]

    # Normalize to percentiles for stability across meshes
    y25, y60 = np.quantile(y, 0.25), np.quantile(y, 0.60)
    y80 = np.quantile(y, 0.80)
    z40 = np.quantile(z, 0.40)

    lobe = np.full(coords.shape[0], "OTHER", dtype=object)
    lobe[y >= y80] = "FRONTAL"
    lobe[(y >= y60) & (y < y80)] = "PARIETAL"
    lobe[(y >= y25) & (y < y60)] = "TEMPORAL"
    lobe[(y < y25) & (z > z40)] = "OCCIPITAL"
    lobe[(y < y25) & (z <= z40)] = "TEMPORAL"
    return lobe


# ============================================================
# OPTIONAL INTERNAL STRUCTURES (schematic) ✅ solves “corpus callosum missing”
# ============================================================

def make_corpus_callosum_schematic(n_lat: int = 18, n_lon: int = 36) -> dict:
    """
    Returns a simple internal 'corpus callosum' schematic as a mesh.
    This is NOT anatomically exact; it's a labeled schematic for UI completeness.
    """
    # A thin "C" / torus-like band approximated by a bent ellipsoid strip
    # We'll just make an ellipsoid-ish sheet centered near midline.
    lat = np.linspace(0.2, np.pi - 0.2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    a, b, c = 12.0, 28.0, 10.0
    coords = []
    for th in lat:
        for ph in lon:
            x = a * np.sin(th) * np.cos(ph) * 0.25  # narrow in x (midline)
            y = b * np.sin(th) * np.sin(ph) * 0.85
            z = c * np.cos(th) * 0.75
            # shift slightly anterior/superior
            y += 5.0
            z += 8.0
            coords.append((x, y, z))
    coords = np.array(coords, dtype=float)

    faces = []
    def vid(i, j): return i * n_lon + (j % n_lon)
    for i in range(n_lat - 1):
        for j in range(n_lon):
            v0 = vid(i, j)
            v1 = vid(i + 1, j)
            v2 = vid(i, j + 1)
            v3 = vid(i + 1, j + 1)
            faces.append((v0, v1, v2))
            faces.append((v2, v1, v3))
    faces = np.array(faces, dtype=int)

    return {"coords": coords, "faces": faces, "label": "corpus_callosum_schematic"}


# ============================================================
# HINDBRAIN (schematic): cerebellum + brainstem ✅ for BioDigital-style demo
# ============================================================

def _make_ellipsoid_mesh(center=(0.0, 0.0, 0.0), radii=(10.0, 10.0, 10.0), n_lat: int = 22, n_lon: int = 44) -> dict:
    """Create a closed ellipsoid mesh (triangulated).

    This is schematic (not anatomically exact) but renders cleanly and is lightweight.
    """
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    rx, ry, rz = (float(radii[0]), float(radii[1]), float(radii[2]))

    lat = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)

    coords = []
    for th in lat:
        for ph in lon:
            x = cx + rx * np.sin(th) * np.cos(ph)
            y = cy + ry * np.sin(th) * np.sin(ph)
            z = cz + rz * np.cos(th)
            coords.append((x, y, z))
    coords = np.asarray(coords, dtype=float)

    faces = []
    def vid(i, j):
        return i * n_lon + (j % n_lon)

    for i in range(n_lat - 1):
        for j in range(n_lon):
            v0 = vid(i, j)
            v1 = vid(i + 1, j)
            v2 = vid(i, j + 1)
            v3 = vid(i + 1, j + 1)
            faces.append((v0, v1, v2))
            faces.append((v2, v1, v3))

    return {"coords": np.asarray(coords, float), "faces": np.asarray(faces, int)}


def make_hindbrain_schematic() -> list[dict]:
    """Return schematic meshes for cerebellum (L/R) and brainstem.

    Coordinates are chosen to sit posterior/inferior relative to fsaverage cortex.
    """
    # NOTE: fsaverage surfaces are roughly centered near (0,0,0) in mm.
    # We place the cerebellum posterior (y<0) and inferior (z<0).

    # Cerebellar hemispheres
    cere_l = _make_ellipsoid_mesh(center=(-18.0, -62.0, -38.0), radii=(22.0, 18.0, 16.0))
    cere_l["label"] = "cerebellum_left_schematic"

    cere_r = _make_ellipsoid_mesh(center=(18.0, -62.0, -38.0), radii=(22.0, 18.0, 16.0))
    cere_r["label"] = "cerebellum_right_schematic"

    # Vermis / midline bridge (small)
    vermis = _make_ellipsoid_mesh(center=(0.0, -60.0, -40.0), radii=(10.0, 14.0, 12.0), n_lat=18, n_lon=36)
    vermis["label"] = "vermis_schematic"

    # Brainstem (tapered-looking via ellipsoid; good enough for demo)
    stem = _make_ellipsoid_mesh(center=(0.0, -40.0, -55.0), radii=(10.0, 14.0, 26.0), n_lat=24, n_lon=40)
    stem["label"] = "brainstem_schematic"

    return [cere_l, cere_r, vermis, stem]



# ============================================================
# MAIN ENTRY: cohesive output for your app
# ============================================================

def load_brain_mesh(
    mesh: str = "fsaverage5",
    n_parcels: int = 120,
    seed: int = 7,
    include_lobes: bool = True,
    include_internal_schematic: bool = True,
) -> dict:
    """
    Cohesive brain mesh loader for the app.

    Returns:
      coords_l, faces_l, labels_l
      coords_r, faces_r, labels_r
      centers_l, centers_r
      lobes_l, lobes_r (optional)
      internal (optional list of schematic meshes)

    Notes:
      - parcel labels are 1..n_parcels for each hemi independently (consistent ID space per hemi).
      - If you want global parcel ids across BOTH hemis, do that in the app by offsetting right hemi.
    """
    m = load_fsaverage_mesh(mesh=mesh)

    coords_l = m["coords_l"]; faces_l = m["faces_l"]
    coords_r = m["coords_r"]; faces_r = m["faces_r"]

    labels_l, centers_l = make_parcels_from_coords(coords_l, n_parcels=n_parcels, seed=seed)
    labels_r, centers_r = make_parcels_from_coords(coords_r, n_parcels=n_parcels, seed=seed + 1)

    out = {
        "coords_l": coords_l, "faces_l": faces_l, "labels_l": labels_l, "centers_l": centers_l,
        "coords_r": coords_r, "faces_r": faces_r, "labels_r": labels_r, "centers_r": centers_r,
        "n_parcels": int(n_parcels),
        "mesh_name": str(mesh),
        "color_hint_cortex": "rgb(210, 190, 160)",  # tan cortex hint (UI can use)
    }

    if include_lobes:
        lobes = get_lobe_labels_fsaverage(mesh=mesh)
        out["lobes_l"] = lobes["lobe_l"]
        out["lobes_r"] = lobes["lobe_r"]
        out["lobe_source"] = lobes.get("atlas_name", "unknown")

    if include_internal_schematic:
        # For the demo viewer: show cerebellum + brainstem (schematic).
        out["internal"] = make_hindbrain_schematic()
    else:
        out["internal"] = []

    return out