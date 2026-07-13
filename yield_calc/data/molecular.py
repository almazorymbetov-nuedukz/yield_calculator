"""Helpers for building lightweight molecular cluster datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def build_molecular_cluster_dataset(
    clusters: Any,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert molecular clusters into a numeric feature matrix.

    The implementation intentionally stays lightweight so it works without RDKit
    or external chemistry packages. It can later be extended with RDKit/Avogadro
    geometries once those dependencies are available.
    """

    if isinstance(clusters, dict):
        cluster_list = clusters.get("clusters", [])
    else:
        cluster_list = list(clusters or [])

    feature_names = [
        "cluster_size",
        "hba_count",
        "hbd_count",
        "biodiesel_count",
        "hydrogen_bond_score",
        "polarity_proxy",
        "miscibility_proxy",
        "component_diversity",
    ]

    rows: List[List[float]] = []
    payload: List[Dict[str, Any]] = []

    for cluster in cluster_list:
        if isinstance(cluster, dict):
            name = str(cluster.get("name", "cluster"))
            components = cluster.get("components", [])
        else:
            name = "cluster"
            components = []

        hba_count = 0
        hbd_count = 0
        biodiesel_count = 0

        for component in components:
            if not isinstance(component, dict):
                continue
            role = str(component.get("role", "")).lower()
            count = int(component.get("count", 0) or 0)
            if role == "hba":
                hba_count += count
            elif role == "hbd":
                hbd_count += count
            elif role in {"biodiesel", "bio", "biodiesel_component"}:
                biodiesel_count += count

        cluster_size = len(components)
        hydrogen_bond_score = 1.5 * hba_count + hbd_count + 0.25 * biodiesel_count
        polarity_proxy = hba_count + 0.5 * hbd_count + 0.2 * biodiesel_count
        miscibility_proxy = 0.7 * hba_count + 0.6 * hbd_count + 0.4 * biodiesel_count
        component_diversity = float(len({str(c.get("name", "")).lower() for c in components if isinstance(c, dict)}))

        row = [
            float(cluster_size),
            float(hba_count),
            float(hbd_count),
            float(biodiesel_count),
            float(hydrogen_bond_score),
            float(polarity_proxy),
            float(miscibility_proxy),
            float(component_diversity),
        ]
        rows.append(row)
        payload.append(
            {
                "name": name,
                "components": components,
                "features": dict(zip(feature_names, row)),
            }
        )

    matrix = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(feature_names)), dtype=np.float32)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(
                {
                    "feature_names": feature_names,
                    "clusters": payload,
                    "feature_matrix": matrix.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "clusters": payload,
        "feature_names": feature_names,
        "feature_matrix": matrix,
    }
