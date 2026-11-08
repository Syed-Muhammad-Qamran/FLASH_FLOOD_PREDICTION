"""3x3 spatial patch helpers for Model A.

Open-Meteo accepts point coordinates; a 3x3 neighbourhood is represented as the
centre plus 8 offsets. Offsets are approximate degrees and are suitable for a
prototype ingest grid, not survey-grade watershed delineation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# ~0.1 degree is ~11 km at the equator; adequate as a coarse convective patch.
DEFAULT_STEP_DEG = 0.10


@dataclass(frozen=True)
class GridPoint:
    row: int
    col: int
    lat: float
    lon: float
    is_center: bool = False


def patch_3x3(lat: float, lon: float, *, step_deg: float = DEFAULT_STEP_DEG) -> list[GridPoint]:
    """Return 9 points in row-major order, centre at (1, 1)."""
    points: list[GridPoint] = []
    for row, dlat in enumerate((-step_deg, 0.0, step_deg)):
        for col, dlon in enumerate((-step_deg, 0.0, step_deg)):
            points.append(
                GridPoint(
                    row=row,
                    col=col,
                    lat=float(lat + dlat),
                    lon=float(lon + dlon),
                    is_center=(row == 1 and col == 1),
                )
            )
    return points


def patch_for_basins(
    basins: Iterable[dict], *, step_deg: float = DEFAULT_STEP_DEG
) -> dict[str, list[GridPoint]]:
    return {
        str(basin["basin_id"]): patch_3x3(
            float(basin["lat"]), float(basin["lon"]), step_deg=step_deg
        )
        for basin in basins
    }
