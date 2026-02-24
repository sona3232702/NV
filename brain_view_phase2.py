"""brain_view_phase2.py
Plotly brain rendering helpers extracted from the working Phase-2 app.

Patch:
- If highlight_lobes is provided and highlight_score is None,
  render a "pending" CYAN overlay so the expected lobe lights up before scoring.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go


def _mesh3d_lighting():
    return dict(ambient=0.55, diffuse=0.95, specular=0.25, roughness=0.7, fresnel=0.15)

def _mesh3d_lightpos():
    return dict(x=1.8, y=1.2, z=0.8)

def _build_mesh_traces(mesh: Dict[str, Any], hemisphere: str = "BOTH") -> List[go.Mesh3d]:
    traces: List[go.Mesh3d] = []

    def add_one(coords, faces, name):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        traces.append(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                name=name,
                opacity=0.98,
                color="rgba(240,120,170,0.78)",
                flatshading=False,
                lighting=_mesh3d_lighting(),
                lightposition=_mesh3d_lightpos(),
                showscale=False,
            )
        )

    hemi = hemisphere.upper()
    if hemi in ("BOTH", "LEFT"):
        add_one(mesh["coords_l"], mesh["faces_l"], "Left")
    if hemi in ("BOTH", "RIGHT"):
        add_one(mesh["coords_r"], mesh["faces_r"], "Right")

    return traces


def _apply_figure_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        ),
        showlegend=False,
    )
    return fig


def _domain_to_lobes(domain: str) -> List[str]:
    d = (domain or "").upper()
    if d == "LANG":
        return ["TEMPORAL", "FRONTAL"]
    if d == "MOT":
        return ["FRONTAL", "PARIETAL"]
    if d == "VIS":
        return ["OCCIPITAL"]
    if d == "AUD":
        return ["TEMPORAL"]
    if d == "EXEC":
        return ["FRONTAL"]
    return []


def _score_to_rgba(score: int) -> str:
    """MRI-like palette: correct=red, partial=amber, incorrect=blue."""
    s = int(score)
    if s >= 2:
        return "rgba(255,60,60,0.85)"   # red
    if s == 1:
        return "rgba(255,180,60,0.82)"  # amber
    return "rgba(60,140,255,0.80)"      # blue

def _pending_rgba() -> str:
    return "rgba(0, 210, 255, 0.70)"    # cyan

def _add_score_overlay_traces(
    fig: go.Figure,
    mesh: Dict[str, Any],
    domain_scores: Dict[str, int],
    hemisphere: str = "BOTH",
) -> None:
    def add_centers(centers, lobe_labels, hemi_name):
        if centers is None or lobe_labels is None:
            return
        centers = np.asarray(centers)
        lobe_labels = np.asarray(lobe_labels)

        lobe_score: Dict[str, int] = {}
        for dom, sc in domain_scores.items():
            for lobe in _domain_to_lobes(dom):
                if lobe not in lobe_score:
                    lobe_score[lobe] = sc
                else:
                    lobe_score[lobe] = min(lobe_score[lobe], sc)

        if not lobe_score:
            return

        for lobe, sc in lobe_score.items():
            mask = (lobe_labels == lobe)
            if mask.sum() == 0:
                continue
            pts = centers[mask]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=6, color=_score_to_rgba(int(sc))),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    hemi = hemisphere.upper()
    if hemi in ("BOTH", "LEFT"):
        add_centers(mesh.get("centers_l"), mesh.get("lobe_labels_l"), "L")
    if hemi in ("BOTH", "RIGHT"):
        add_centers(mesh.get("centers_r"), mesh.get("lobe_labels_r"), "R")


def _add_lobe_overlay_traces(
    fig: "go.Figure",
    mesh: Dict[str, Any],
    lobes: List[str],
    *,
    rgba: str,
    hemisphere: str = "BOTH",
) -> None:
    if not lobes:
        return
    lobes = [str(x).upper() for x in lobes]

    def _mask_from_lobes(arr: np.ndarray) -> np.ndarray:
        arr_u = np.array([str(x).upper() for x in arr], dtype=object)
        return np.isin(arr_u, lobes).astype(float)

    colorscale = [
        [0.0, "rgba(0,0,0,0)"],
        [0.01, "rgba(0,0,0,0)"],
        [1.0, rgba],
    ]

    hemi = hemisphere.upper()
    if hemi in ("BOTH", "LEFT"):
        lobes_l = np.asarray(mesh.get("lobes_l", []))
        if lobes_l.size:
            fig.add_trace(go.Mesh3d(
                x=mesh["coords_l"][:,0], y=mesh["coords_l"][:,1], z=mesh["coords_l"][:,2],
                i=mesh["faces_l"][:,0], j=mesh["faces_l"][:,1], k=mesh["faces_l"][:,2],
                intensity=_mask_from_lobes(lobes_l),
                colorscale=colorscale,
                opacity=0.75,
                hoverinfo="skip",
                showscale=False,
                lighting=_mesh3d_lighting(),
                lightposition=_mesh3d_lightpos(),
                name="Lobe overlay L",
                showlegend=False,
            ))

    if hemi in ("BOTH", "RIGHT"):
        lobes_r = np.asarray(mesh.get("lobes_r", []))
        if lobes_r.size:
            fig.add_trace(go.Mesh3d(
                x=mesh["coords_r"][:,0], y=mesh["coords_r"][:,1], z=mesh["coords_r"][:,2],
                i=mesh["faces_r"][:,0], j=mesh["faces_r"][:,1], k=mesh["faces_r"][:,2],
                intensity=_mask_from_lobes(lobes_r),
                colorscale=colorscale,
                opacity=0.75,
                hoverinfo="skip",
                showscale=False,
                lighting=_mesh3d_lighting(),
                lightposition=_mesh3d_lightpos(),
                name="Lobe overlay R",
                showlegend=False,
            ))

def make_brain_figure(
    mesh: Dict[str, Any],
    *,
    hemisphere: str = "BOTH",
    domain_scores: Optional[Dict[str, int]] = None,
    highlight_lobes: Optional[List[str]] = None,
    highlight_score: Optional[int] = None,
) -> "go.Figure":
    fig = go.Figure()

    for tr in _build_mesh_traces(mesh, hemisphere=hemisphere):
        fig.add_trace(tr)

    if domain_scores:
        _add_score_overlay_traces(fig, mesh, domain_scores, hemisphere=hemisphere)

    if highlight_lobes:
        if highlight_score is None:
            rgba = _pending_rgba()
        else:
            rgba = _score_to_rgba(int(highlight_score))
        _add_lobe_overlay_traces(fig, mesh, list(highlight_lobes), rgba=rgba, hemisphere=hemisphere)

    return _apply_figure_layout(fig)
