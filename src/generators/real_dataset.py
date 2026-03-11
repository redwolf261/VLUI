"""
Real-dataset loader for spatial index benchmarking.

Supported formats
-----------------
Porto Taxi (Kaggle ECML/PKDD 2015)
    CSV columns: TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID,
                 TIMESTAMP, DAY_TYPE, MISSING_DATA, POLYLINE
    The POLYLINE column is a JSON array of [lon, lat] pairs.
    We extract the trip *origin* (first coordinate of each polyline) as the
    insert point, and the trip *destination* (last coordinate) as the query
    point (range query centred on destination).

Generic lon/lat CSV
    Any CSV with numeric longitude and latitude columns (auto-detected).
    Column name heuristics: lon/lng/longitude, lat/latitude, x/y.

NYC Taxi (TLC yellow/green trip records, Parquet or CSV)
    Columns: pickup_longitude, pickup_latitude (pre-2016 CSVs)
             or pickup_location_id (post-2016, not usable without geometry).

OpenStreetMap POI extract (CSV exported from Overpass or osmconvert)
    Columns: id, lat, lon, ...

Usage
-----
    loader = RealDatasetLoader.from_porto_csv("train.csv",
                                              n_points=5000,
                                              n_queries=1000,
                                              grid_dims=(2, 2))
    trace = loader.generate()

    loader = RealDatasetLoader.from_csv("mydata.csv",
                                        lon_col="longitude",
                                        lat_col="latitude",
                                        n_points=5000,
                                        n_queries=1000)
    trace = loader.generate()
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.models.spatial import (
    OpType,
    Operation,
    SpatialPoint,
    SpatialRegion,
    WorkloadTrace,
    assign_region,
    build_grid_partition,
)


# ── coordinate pair ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


# ── bounding boxes for well-known cities (used for auto-clipping) ──────────────

CITY_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    # (lon_min, lat_min, lon_max, lat_max)
    "porto":   (-8.75,  41.10, -8.55, 41.25),
    "nyc":     (-74.05,  40.63, -73.75, 40.88),
    "beijing": (116.20,  39.75, 116.55, 40.05),
    "global":  (-180.0, -90.0,  180.0,  90.0),
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _try_parse_polyline(raw: str) -> list[LatLon] | None:
    """Parse a Porto-style POLYLINE field like '[[-8.61,41.14],[-8.62,41.15]]'."""
    raw = raw.strip()
    if not raw or raw in ("", "[]", "false", "False"):
        return None
    try:
        pairs = json.loads(raw)
        if not isinstance(pairs, list) or len(pairs) < 2:
            return None
        return [LatLon(lat=float(p[1]), lon=float(p[0])) for p in pairs]
    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
        return None


def _detect_lon_lat_cols(header: list[str]) -> tuple[str, str]:
    """Return (lon_col, lat_col) guessing from header names."""
    header_lower = [h.lower().strip() for h in header]

    lon_candidates = ["lon", "lng", "longitude", "pickup_longitude", "x", "long"]
    lat_candidates = ["lat", "latitude",  "pickup_latitude",  "y"]

    lon_col = lat_col = ""
    for cand in lon_candidates:
        if cand in header_lower:
            lon_col = header[header_lower.index(cand)]
            break
    for cand in lat_candidates:
        if cand in header_lower:
            lat_col = header[header_lower.index(cand)]
            break

    if not lon_col or not lat_col:
        raise ValueError(
            f"Could not auto-detect lon/lat columns from header: {header}. "
            "Pass lon_col and lat_col explicitly."
        )
    return lon_col, lat_col


def _clip_and_normalize(
    coords: list[tuple[float, float]],  # (lon, lat) pairs
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Clip to bounding box and normalize to [0, 1]²."""
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    result = []
    for lon, lat in coords:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            result.append((
                (lon - lon_min) / lon_span,
                (lat - lat_min) / lat_span,
            ))
    return result


def _auto_bounds(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Compute tight bounding box with 2% padding."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lon_span = max(lons) - min(lons) or 1.0
    lat_span = max(lats) - min(lats) or 1.0
    pad_lon = lon_span * 0.02
    pad_lat = lat_span * 0.02
    return (
        min(lons) - pad_lon,
        min(lats) - pad_lat,
        max(lons) + pad_lon,
        max(lats) + pad_lat,
    )


# ── main loader ────────────────────────────────────────────────────────────────

class RealDatasetLoader:
    """Loads real-world spatial coordinates and builds a WorkloadTrace.

    Parameters
    ----------
    coords      : normalized (x, y) pairs in [0,1]², one per insert point
    query_radius: half-width of the range-query window (in normalized space)
    n_points    : how many insert points to use (sampled without replacement)
    n_queries   : how many range queries to interleave
    grid_dims   : partition grid shape
    query_frac  : fraction of ops that are queries (ignored if n_queries given)
    seed        : random seed for query placement interleaving
    source_name : human-readable label for this dataset
    bounds      : original (lon_min, lat_min, lon_max, lat_max) for metadata
    """

    def __init__(
        self,
        coords: list[tuple[float, float]],
        *,
        query_radius: float = 0.05,
        n_points: int | None = None,
        n_queries: int | None = None,
        grid_dims: tuple[int, int] = (2, 2),
        seed: int = 42,
        source_name: str = "real_dataset",
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ):
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Sample / cap insert points
        if n_points is not None and n_points < len(coords):
            coords = rng.sample(coords, n_points)
        self._coords = coords
        self._n_points = len(coords)
        self._n_queries = n_queries if n_queries is not None else max(1, self._n_points // 5)
        self._query_radius = query_radius
        self._grid_dims = grid_dims
        self._seed = seed
        self._np_rng = np_rng
        self._source_name = source_name
        self._bounds = bounds

        lo = np.zeros(2)
        hi = np.ones(2)
        self._regions = build_grid_partition(lo, hi, grid_dims)

    # ── factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_porto_csv(
        cls,
        path: str | Path | io.IOBase,
        *,
        n_points: int = 5000,
        n_queries: int = 1000,
        grid_dims: tuple[int, int] = (2, 2),
        query_radius: float = 0.04,
        seed: int = 42,
        city_bounds: tuple[float, float, float, float] | None = None,
    ) -> "RealDatasetLoader":
        """Load Porto Taxi dataset (Kaggle ECML/PKDD 2015).

        The CSV must have a POLYLINE column containing JSON arrays of
        [lon, lat] coordinate pairs. Each non-empty polyline contributes
        its origin point as an insert and its destination as a query centre.

        Args:
            path: path to train.csv (or any compatible file-like object)
            n_points: max insert points to use
            n_queries: number of range queries to generate
            grid_dims: partition grid (default 2×2 matching benchmark)
            query_radius: range-query half-width in normalized space
            seed: reproducibility seed
            city_bounds: (lon_min, lat_min, lon_max, lat_max) — defaults to Porto
        """
        bounds = city_bounds or CITY_BOUNDS["porto"]
        raw_coords: list[tuple[float, float]] = []

        if isinstance(path, (str, Path)):
            fh = open(path, newline="", encoding="utf-8")
            close_after = True
        else:
            fh = path  # file-like (e.g. from Flask request.files)
            close_after = False

        try:
            reader = csv.DictReader(fh)
            for row in reader:
                poly = _try_parse_polyline(row.get("POLYLINE", ""))
                if poly is None:
                    continue
                origin = poly[0]
                raw_coords.append((origin.lon, origin.lat))
                if len(raw_coords) >= n_points * 10:   # read ahead buffer
                    break
        finally:
            if close_after:
                fh.close()

        coords_norm = _clip_and_normalize(raw_coords, bounds)
        if len(coords_norm) < n_points:
            raise ValueError(
                f"Only {len(coords_norm)} valid Porto coordinates found in "
                f"bounds {bounds}. Need at least {n_points}. "
                f"Check the file or supply different city_bounds."
            )

        return cls(
            coords_norm,
            n_points=n_points,
            n_queries=n_queries,
            grid_dims=grid_dims,
            query_radius=query_radius,
            seed=seed,
            source_name=f"porto_taxi (n={n_points})",
            bounds=bounds,
        )

    @classmethod
    def from_nyc_taxi_csv(
        cls,
        path: str | Path | io.IOBase,
        *,
        n_points: int = 5000,
        n_queries: int = 1000,
        grid_dims: tuple[int, int] = (2, 2),
        query_radius: float = 0.04,
        seed: int = 42,
        city_bounds: tuple[float, float, float, float] | None = None,
    ) -> "RealDatasetLoader":
        """Load NYC TLC yellow taxi dataset (pre-2016 CSV format).

        Expects columns: pickup_longitude, pickup_latitude
        Recommended file: yellow_tripdata_2015-01.csv from the TLC Trip Record Data page.

        Args:
            path: path to CSV file (or file-like object)
            n_points: max insert points to use
            n_queries: number of range queries to generate
            grid_dims: partition grid (default 2×2)
            query_radius: range-query half-width in normalized space
            seed: reproducibility seed
            city_bounds: (lon_min, lat_min, lon_max, lat_max) — defaults to NYC
        """
        bounds = city_bounds or CITY_BOUNDS["nyc"]
        raw_coords: list[tuple[float, float]] = []

        if isinstance(path, (str, Path)):
            fh = open(path, newline="", encoding="utf-8")
            close_after = True
        else:
            fh = path
            close_after = False

        try:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            lon_col, lat_col = _detect_lon_lat_cols(list(header))

            for row in reader:
                try:
                    lon = float(row[lon_col])
                    lat = float(row[lat_col])
                    if math.isfinite(lon) and math.isfinite(lat):
                        raw_coords.append((lon, lat))
                except (KeyError, ValueError):
                    continue
                if len(raw_coords) >= n_points * 10:
                    break
        finally:
            if close_after:
                fh.close()

        coords_norm = _clip_and_normalize(raw_coords, bounds)
        if len(coords_norm) < n_points:
            raise ValueError(
                f"Only {len(coords_norm)} valid NYC coordinates found in "
                f"bounds {bounds}. Need at least {n_points}. "
                f"Check the file, or supply a different city_bounds."
            )

        return cls(
            coords_norm,
            n_points=n_points,
            n_queries=n_queries,
            grid_dims=grid_dims,
            query_radius=query_radius,
            seed=seed,
            source_name=f"nyc_taxi (n={n_points})",
            bounds=bounds,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path | io.IOBase,
        *,
        lon_col: str = "",
        lat_col: str = "",
        n_points: int = 5000,
        n_queries: int = 1000,
        grid_dims: tuple[int, int] = (2, 2),
        query_radius: float = 0.04,
        seed: int = 42,
        city_bounds: tuple[float, float, float, float] | None = None,
        skip_header: bool = True,
        delimiter: str = ",",
    ) -> "RealDatasetLoader":
        """Load any lon/lat CSV.

        If lon_col/lat_col not supplied, header is inspected for common names
        (lon, lng, longitude, x, lat, latitude, y).

        If city_bounds not provided, tight bounds are computed from the data.
        """
        raw_coords: list[tuple[float, float]] = []

        if isinstance(path, (str, Path)):
            fh = open(path, newline="", encoding="utf-8")
            close_after = True
        else:
            fh = path
            close_after = False

        try:
            reader = csv.DictReader(fh, delimiter=delimiter)
            header = reader.fieldnames or []
            if not lon_col or not lat_col:
                lon_col, lat_col = _detect_lon_lat_cols(list(header))

            for row in reader:
                try:
                    lon = float(row[lon_col])
                    lat = float(row[lat_col])
                    if math.isfinite(lon) and math.isfinite(lat):
                        raw_coords.append((lon, lat))
                except (KeyError, ValueError):
                    continue
                if len(raw_coords) >= n_points * 10:
                    break
        finally:
            if close_after:
                fh.close()

        if len(raw_coords) < n_points:
            raise ValueError(
                f"Only {len(raw_coords)} valid rows found in CSV. Need at least {n_points}."
            )

        bounds = city_bounds or _auto_bounds(raw_coords)
        coords_norm = _clip_and_normalize(raw_coords, bounds)

        return cls(
            coords_norm,
            n_points=n_points,
            n_queries=n_queries,
            grid_dims=grid_dims,
            query_radius=query_radius,
            seed=seed,
            source_name=f"csv ({Path(path).name if isinstance(path, (str, Path)) else 'upload'}, n={n_points})",
            bounds=bounds,
        )

    # ── trace generation ───────────────────────────────────────────────────────

    def generate(self) -> WorkloadTrace:
        """Convert loaded coordinates into a WorkloadTrace.

        Insert ops come first (in coordinate order), interleaved with
        uniformly-spaced range queries so the CRI measurer sees a
        realistic mix throughout the trace.
        """
        lo = np.zeros(2)
        hi = np.ones(2)
        regions = self._regions

        total = self._n_points + self._n_queries
        query_timestamps = set(
            self._np_rng.choice(total, size=self._n_queries, replace=False).tolist()
        )

        trace = WorkloadTrace(
            regions=list(regions),
            metadata={
                "generator": "RealDatasetLoader",
                "source": self._source_name,
                "n_points": self._n_points,
                "n_queries": self._n_queries,
                "grid_dims": list(self._grid_dims),
                "query_radius": self._query_radius,
                "seed": self._seed,
                "bounds": list(self._bounds),
            },
        )

        coord_iter = iter(self._coords)
        point_id = 0

        for t in range(total):
            if t in query_timestamps:
                # Range query: pick a random region centre
                qrid = int(self._np_rng.integers(0, len(regions)))
                qreg = regions[qrid]
                center = (qreg.lo + qreg.hi) / 2.0
                r = self._query_radius
                qbox = SpatialRegion(
                    lo=np.maximum(center - r, lo),
                    hi=np.minimum(center + r, hi),
                    region_id=qrid,
                )
                trace.append(Operation(
                    op_type=OpType.RANGE_QUERY,
                    timestamp=t,
                    query_region=qbox,
                    region_id=qrid,
                ))
            else:
                try:
                    x, y = next(coord_iter)
                except StopIteration:
                    break
                coords_arr = np.array([x, y], dtype=np.float64)
                rid = assign_region(
                    SpatialPoint(coords=coords_arr, point_id=point_id),
                    regions,
                )
                if rid < 0:
                    rid = 0  # clamp out-of-bounds (shouldn't happen after normalization)
                pt = SpatialPoint(coords=coords_arr, point_id=point_id, timestamp=t)
                trace.append(Operation(
                    op_type=OpType.INSERT,
                    timestamp=t,
                    point=pt,
                    region_id=rid,
                ))
                point_id += 1

        return trace

    # ── metadata ───────────────────────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def region_point_counts(self) -> dict[int, int]:
        """Count how many insert points fall in each region (skew diagnostic)."""
        from src.models.spatial import SpatialPoint
        counts: dict[int, int] = {r.region_id: 0 for r in self._regions}
        for x, y in self._coords:
            coords_arr = np.array([x, y], dtype=np.float64)
            rid = assign_region(
                SpatialPoint(coords=coords_arr, point_id=0),
                self._regions,
            )
            if rid >= 0:
                counts[rid] = counts.get(rid, 0) + 1
        return counts
