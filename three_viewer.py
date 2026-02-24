from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


def _to_list(a) -> list:
    if a is None:
        return []
    if isinstance(a, list):
        return a
    if isinstance(a, np.ndarray):
        return a.tolist()
    return list(a)


def render_three_viewer(
    *,
    mesh: Dict[str, Any],
    intensity_by_vertex_l: Optional[np.ndarray] = None,
    intensity_by_vertex_r: Optional[np.ndarray] = None,
    show_overlay: bool = True,
    overlay_opacity: float = 0.28,
    base_color: str = "#d8c7b0",
    hemisphere_view: str = "BOTH",
    show_internal: bool = False,
    show_subcortical: bool = False,
    show_vessels: bool = False,
    height: int = 680,
) -> None:
    """Render cortex + optional layers using Three.js inside Streamlit.

    Notes:
    - Uses CDN for three.js + OrbitControls (works in most Streamlit/Kaggle setups).
    - Keeps UI toggles in Streamlit; JS only renders what you pass.
    """

    coords_l = np.asarray(mesh["coords_l"], dtype=float)
    faces_l = np.asarray(mesh["faces_l"], dtype=np.int32)
    coords_r = np.asarray(mesh["coords_r"], dtype=float)
    faces_r = np.asarray(mesh["faces_r"], dtype=np.int32)

    if intensity_by_vertex_l is None:
        intensity_by_vertex_l = np.zeros(len(coords_l), dtype=float)
    if intensity_by_vertex_r is None:
        intensity_by_vertex_r = np.zeros(len(coords_r), dtype=float)

    show_left = hemisphere_view in ("BOTH", "LEFT")
    show_right = hemisphere_view in ("BOTH", "RIGHT")

    payload: Dict[str, Any] = {
        "baseColor": base_color,
        "overlayOpacity": float(np.clip(overlay_opacity, 0.0, 1.0)),
        "show": {
            "left": bool(show_left),
            "right": bool(show_right),
            "overlay": bool(show_overlay),
            "internal": bool(show_internal),
            "subcortical": bool(show_subcortical),
            "vessels": bool(show_vessels),
        },
        "left": {
            "pos": _to_list(coords_l.reshape(-1)),
            "tri": _to_list(faces_l.reshape(-1)),
            "int": _to_list(np.clip(intensity_by_vertex_l, 0.0, 1.0)),
        },
        "right": {
            "pos": _to_list(coords_r.reshape(-1)),
            "tri": _to_list(faces_r.reshape(-1)),
            "int": _to_list(np.clip(intensity_by_vertex_r, 0.0, 1.0)),
        },
        "layers": {
            "internal": [
                {
                    "label": str(m.get("label", "internal")),
                    "pos": _to_list(np.asarray(m.get("coords", []), float).reshape(-1)),
                    "tri": _to_list(np.asarray(m.get("faces", []), np.int32).reshape(-1)),
                }
                for m in (mesh.get("internal") or mesh.get("internal_schematic") or mesh.get("schematic") or [])
            ],
            "subcortical": [
                {
                    "label": str(m.get("label", f"sub_{i}")),
                    "pos": _to_list(np.asarray(m.get("coords", []), float).reshape(-1)),
                    "tri": _to_list(np.asarray(m.get("faces", []), np.int32).reshape(-1)),
                }
                for i, m in enumerate(mesh.get("subcortical") or [])
            ],
            "vessels": [
                {
                    "label": str(m.get("label", f"vessels_{i}")),
                    "pos": _to_list(np.asarray(m.get("coords", []), float).reshape(-1)),
                    "tri": _to_list(np.asarray(m.get("faces", []), np.int32).reshape(-1)),
                }
                for i, m in enumerate(mesh.get("vessels") or [])
            ],
        },
    }

    # Keep this compact-ish (Streamlit message size can matter)
    data_json = json.dumps(payload, separators=(",", ":"))

    # A small, nice colormap (Plasma-ish) as a JS function.
    # (Avoid heavy deps; this is close enough visually.)
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      html, body {{ margin: 0; padding: 0; height: 100%; background: #0b0f14; }}
      #wrap {{ position: relative; width: 100%; height: 100%; }}
      #tip {{
        position: absolute; left: 12px; bottom: 10px;
        color: rgba(255,255,255,0.72);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        font-size: 12px; user-select: none;
        background: rgba(0,0,0,0.25); padding: 6px 8px; border-radius: 10px;
        backdrop-filter: blur(6px);
      }}
      canvas {{ display: block; }}
    </style>
  </head>
  <body>
    <div id="wrap">
      <div id="tip">Drag to rotate • Scroll to zoom • Right-drag to pan</div>
    </div>

    <script>
      window.__MESH_DATA__ = {data_json};
    </script>

    <script>
      // Helpful debug: surface JS errors inside the iframe.
      window.onerror = function(message, source, lineno, colno, error) {
        const tip = document.getElementById('tip');
        if (tip) {
          tip.textContent = 'Viewer error: ' + message + ' (open DevTools console for details)';
          tip.style.background = 'rgba(180,30,30,0.35)';
        }
      };
    </script>

    <!--
      Use ES modules for broad compatibility with modern three.js builds.
      Some CDNs no longer ship the legacy examples/js/* scripts.
    -->
    <script type="module">
      import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
      import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

      const data = window.__MESH_DATA__;
      const wrap = document.getElementById('wrap');

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0f14);

      const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 4000);
      camera.position.set(280, 120, 260);

      const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.0;
      wrap.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.screenSpacePanning = true;

      // Lights (BioDigital-ish)
      scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.85));
      const dir = new THREE.DirectionalLight(0xffffff, 1.05);
      dir.position.set(220, 240, 180);
      scene.add(dir);
      const rim = new THREE.DirectionalLight(0x88aaff, 0.35);
      rim.position.set(-250, 120, -220);
      scene.add(rim);

      function hexToColor(hex) {{
        const c = new THREE.Color(hex);
        return c;
      }}

      // Plasma-ish colormap (t in [0,1])
      function plasma(t) {{
        t = Math.max(0, Math.min(1, t));
        // Coeff fit (approx). Produces nice purple->yellow.
        const r = 0.050 + 0.950 * Math.pow(t, 0.8);
        const g = 0.030 + 0.920 * Math.pow(t, 1.2);
        const b = 0.160 + 0.840 * Math.pow(1.0 - t, 0.55);
        return new THREE.Color(r, g, b);
      }}

      function buildMesh(layer, material, vertexColors=null) {{
        if (!layer || !layer.pos || layer.pos.length === 0 || !layer.tri || layer.tri.length === 0) return null;
        const geom = new THREE.BufferGeometry();
        const pos = new Float32Array(layer.pos);
        geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const idx = (pos.length/3 > 65535)
          ? new Uint32Array(layer.tri)
          : new Uint16Array(layer.tri);
        geom.setIndex(new THREE.BufferAttribute(idx, 1));
        geom.computeVertexNormals();
        if (vertexColors) {{
          geom.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3));
        }}
        const m = new THREE.Mesh(geom, material);
        return m;
      }}

      // Cortex materials
      const cortexMat = new THREE.MeshStandardMaterial({{
        color: hexToColor(data.baseColor),
        roughness: 0.72,
        metalness: 0.02,
        transparent: true,
        opacity: (data.show.internal || data.show.subcortical || data.show.vessels) ? 0.28 : 1.0,
        side: THREE.DoubleSide,
      }});

      const overlayMat = new THREE.MeshStandardMaterial({{
        transparent: true,
        opacity: data.overlayOpacity,
        vertexColors: true,
        roughness: 0.85,
        metalness: 0.0,
        depthWrite: false,
        side: THREE.DoubleSide,
      }});

      function buildOverlayColors(intensities) {{
        const n = intensities.length;
        const cols = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {{
          const c = plasma(intensities[i]);
          cols[i*3+0] = c.r;
          cols[i*3+1] = c.g;
          cols[i*3+2] = c.b;
        }}
        return cols;
      }}

      const group = new THREE.Group();
      scene.add(group);

      // Left / Right cortex
      if (data.show.left) {{
        const L = buildMesh(data.left, cortexMat);
        if (L) group.add(L);
        if (data.show.overlay) {{
          const cols = buildOverlayColors(data.left.int || []);
          const OL = buildMesh(data.left, overlayMat, cols);
          if (OL) group.add(OL);
        }}
      }}
      if (data.show.right) {{
        const R = buildMesh(data.right, cortexMat);
        if (R) group.add(R);
        if (data.show.overlay) {{
          const cols = buildOverlayColors(data.right.int || []);
          const OR = buildMesh(data.right, overlayMat, cols);
          if (OR) group.add(OR);
        }}
      }}

      // Internal / Subcortical / Vessels materials
      const internalMat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color(0x9ab4ff),
        transparent: true,
        opacity: 0.55,
        roughness: 0.45,
        metalness: 0.10,
        side: THREE.DoubleSide,
      }});
      const subMatA = new THREE.MeshStandardMaterial({{ color: new THREE.Color(0xeef2ff), transparent: true, opacity: 0.55, roughness: 0.55, metalness: 0.06, side: THREE.DoubleSide }});
      const subMatB = new THREE.MeshStandardMaterial({{ color: new THREE.Color(0xd8f3dc), transparent: true, opacity: 0.50, roughness: 0.55, metalness: 0.06, side: THREE.DoubleSide }});
      const subMatC = new THREE.MeshStandardMaterial({{ color: new THREE.Color(0xffe5d9), transparent: true, opacity: 0.50, roughness: 0.55, metalness: 0.06, side: THREE.DoubleSide }});
      const vesselMat = new THREE.MeshStandardMaterial({{ color: new THREE.Color(0xc94b4b), transparent: true, opacity: 0.96, roughness: 0.35, metalness: 0.18, side: THREE.DoubleSide }});

      if (data.show.internal) {{
        (data.layers.internal || []).forEach((m, i) => {{
          const obj = buildMesh(m, internalMat);
          if (obj) group.add(obj);
        }});
      }}
      if (data.show.subcortical) {{
        const mats = [subMatA, subMatB, subMatC];
        (data.layers.subcortical || []).forEach((m, i) => {{
          const obj = buildMesh(m, mats[i % mats.length]);
          if (obj) group.add(obj);
        }});
      }}
      if (data.show.vessels) {{
        (data.layers.vessels || []).forEach((m, i) => {{
          const obj = buildMesh(m, vesselMat);
          if (obj) group.add(obj);
        }});
      }}

      // Center the group
      const box = new THREE.Box3().setFromObject(group);
      const center = box.getCenter(new THREE.Vector3());
      group.position.sub(center);

      // Fit camera
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fitDist = maxDim * 1.35;
      camera.position.set(fitDist, fitDist * 0.55, fitDist);
      controls.target.set(0, 0, 0);
      controls.update();

      function resize() {{
        const w = wrap.clientWidth;
        const h = wrap.clientHeight;
        camera.aspect = w / Math.max(1, h);
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
      }}

      const ro = new ResizeObserver(resize);
      ro.observe(wrap);
      resize();

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    </script>
  </body>
</html>
"""

    # Use scrolling=False to avoid scroll capture issues.
    components.html(html, height=height, scrolling=False)
