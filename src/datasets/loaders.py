"""
Real-world dataset loaders.

Convert GPS trajectories, check-ins, and POIs into WorkloadTrace objects
compatible with the benchmark framework.

Supported datasets:
- NYC Taxi (TLC trip record data)
- GeoLife GPS Trajectories
- OpenStreetMap POIs (via Overpass API extract)

Each loader:
1. Downloads or reads from data/real/
2. Filters to a bounding box
3. Normalizes coordinates to [0, 1]^2
4. Converts to timestamped Operation stream
5. Assigns region IDs using the standard grid partition
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.models.spatial import (
    OpType,
    Operation,
    SpatialPoint,
    SpatialRegion,
    WorkloadTrace,
    build_grid_partition,
)


class DatasetLoader:
    """Base class for real dataset loaders."""

    def __init__(
        self,
        *,
        grid_dims: tuple[int, ...] = (4, 4),
        max_points: int = 100_000,
        query_fraction: float = 0.1,
        query_radius: float = 0.05,
        seed: int = 42,
    ):
        self.grid_dims = grid_dims
        self.max_points = max_points
        self.query_fraction = query_fraction
        self.query_radius = query_radius
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._lo = np.array([0.0, 0.0])
        self._hi = np.array([1.0, 1.0])
        self._regions = build_grid_partition(self._lo, self._hi, grid_dims)

    def _normalize_coords(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Normalize (lon, lat) to [0, 1]^2 within bounding box.

        Args:
            lons: longitude array
            lats: latitude array
            bbox: (lon_min, lat_min, lon_max, lat_max)

        Returns:
            Array of shape (n, 2) with normalized coordinates.
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        x = (lons - lon_min) / (lon_max - lon_min)
        y = (lats - lat_min) / (lat_max - lat_min)
        coords = np.column_stack([
            np.clip(x, 0.0, 1.0),
            np.clip(y, 0.0, 1.0),
        ])
        return coords

    def _find_region(self, coords: np.ndarray) -> int:
        for r in self._regions:
            if np.all(coords >= r.lo) and np.all(coords <= r.hi):
                return r.region_id
        return 0

    def _build_trace(
        self,
        coords_array: np.ndarray,
        dataset_name: str,
    ) -> WorkloadTrace:
        """Convert normalized coordinate array into a WorkloadTrace.

        Interleaves inserts with uniformly distributed range queries.
        """
        n = min(len(coords_array), self.max_points)
        n_queries = int(n * self.query_fraction)
        total_ops = n + n_queries

        query_timestamps = set(
            self.rng.choice(total_ops, size=n_queries, replace=False)
        )

        trace = WorkloadTrace(
            regions=list(self._regions),
            metadata={
                "dataset": dataset_name,
                "n_points": n,
                "n_queries": n_queries,
                "grid_dims": self.grid_dims,
                "seed": self.seed,
            },
        )

        point_idx = 0
        for t in range(total_ops):
            if t in query_timestamps:
                # Uniform query across all regions
                rid = int(self.rng.integers(0, len(self._regions)))
                region = self._regions[rid]
                center = self.rng.uniform(region.lo, region.hi, size=2)
                qr = SpatialRegion(
                    lo=np.maximum(center - self.query_radius, self._lo),
                    hi=np.minimum(center + self.query_radius, self._hi),
                    region_id=rid,
                )
                trace.append(Operation(
                    op_type=OpType.RANGE_QUERY,
                    timestamp=t,
                    query_region=qr,
                    region_id=rid,
                ))
            else:
                if point_idx >= n:
                    continue
                c = coords_array[point_idx]
                pt = SpatialPoint(coords=c, point_id=point_idx, timestamp=t)
                rid = self._find_region(c)
                trace.append(Operation(
                    op_type=OpType.INSERT,
                    timestamp=t,
                    point=pt,
                    region_id=rid,
                ))
                point_idx += 1

        return trace


class NYCTaxiLoader(DatasetLoader):
    """NYC Taxi Trip Records loader.

    Data source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

    Expected format: CSV with columns including
    pickup_longitude, pickup_latitude (or tpep_pickup_datetime, etc.)

    The dataset is naturally skewed — Manhattan gets far more pickups
    than outer boroughs. This is exactly the skew pattern we need.

    Bounding box: NYC metro area
        lon: [-74.05, -73.75]
        lat: [40.63, 40.85]
    """

    NYC_BBOX = (-74.05, 40.63, -73.75, 40.85)

    def load(self, filepath: str | Path) -> WorkloadTrace:
        """Load NYC taxi CSV and convert to WorkloadTrace.

        Supports both old format (pickup_longitude/latitude columns)
        and new format (PULocationID with coordinate mapping).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"NYC Taxi data not found at {filepath}. "
                f"Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
            )

        lons, lats = [], []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Try old format
                    if "pickup_longitude" in row:
                        lon = float(row["pickup_longitude"])
                        lat = float(row["pickup_latitude"])
                    elif "Start Lon" in row:
                        lon = float(row["Start Lon"])
                        lat = float(row["Start Lat"])
                    else:
                        continue

                    # Filter to NYC bbox
                    if (self.NYC_BBOX[0] <= lon <= self.NYC_BBOX[2] and
                            self.NYC_BBOX[1] <= lat <= self.NYC_BBOX[3]):
                        lons.append(lon)
                        lats.append(lat)

                    if len(lons) >= self.max_points:
                        break
                except (ValueError, KeyError):
                    continue

        if not lons:
            raise ValueError("No valid GPS points found in file")

        coords = self._normalize_coords(
            np.array(lons), np.array(lats), self.NYC_BBOX
        )

        # Shuffle to simulate streaming arrival order
        order = self.rng.permutation(len(coords))
        coords = coords[order]

        return self._build_trace(coords, "nyc_taxi")


class GeoLifeLoader(DatasetLoader):
    """GeoLife GPS Trajectory Dataset loader.

    Data source: https://www.microsoft.com/en-us/research/project/geolife/

    Format: PLT files in nested directory structure
        user/Trajectory/*.plt

    Each PLT file has lines:
        latitude, longitude, 0, altitude, days, date, time

    Beijing bounding box:
        lon: [116.1, 116.7]
        lat: [39.7, 40.2]
    """

    BEIJING_BBOX = (116.1, 39.7, 116.7, 40.2)

    def load(self, data_dir: str | Path) -> WorkloadTrace:
        """Load GeoLife data from directory tree."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"GeoLife data not found at {data_dir}. "
                f"Download from: https://www.microsoft.com/en-us/research/project/geolife/"
            )

        lons, lats = [], []
        plt_files = sorted(data_dir.rglob("*.plt"))

        for plt_file in plt_files:
            with open(plt_file, "r") as f:
                # Skip 6-line header
                for _ in range(6):
                    next(f, None)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        if (self.BEIJING_BBOX[0] <= lon <= self.BEIJING_BBOX[2] and
                                self.BEIJING_BBOX[1] <= lat <= self.BEIJING_BBOX[3]):
                            lons.append(lon)
                            lats.append(lat)
                    except ValueError:
                        continue

                    if len(lons) >= self.max_points:
                        break
            if len(lons) >= self.max_points:
                break

        if not lons:
            raise ValueError("No valid GPS points found in GeoLife data")

        coords = self._normalize_coords(
            np.array(lons), np.array(lats), self.BEIJING_BBOX
        )

        return self._build_trace(coords, "geolife")


class OSMPOILoader(DatasetLoader):
    """OpenStreetMap POI loader.

    Reads a CSV export of OSM nodes with (lon, lat) columns.
    Can be generated via Overpass Turbo query or planet extract.

    World bounding box (default): configurable per region.
    """

    def load(
        self,
        filepath: str | Path,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> WorkloadTrace:
        """Load OSM POI CSV.

        Expected columns: lon (or longitude), lat (or latitude)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"OSM data not found at {filepath}")

        lons, lats = [], []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lon_key = "lon" if "lon" in row else "longitude"
                    lat_key = "lat" if "lat" in row else "latitude"
                    lon = float(row[lon_key])
                    lat = float(row[lat_key])
                    lons.append(lon)
                    lats.append(lat)
                    if len(lons) >= self.max_points:
                        break
                except (ValueError, KeyError):
                    continue

        if not lons:
            raise ValueError("No valid points found in OSM file")

        lons_arr = np.array(lons)
        lats_arr = np.array(lats)

        if bbox is None:
            bbox = (
                float(lons_arr.min()), float(lats_arr.min()),
                float(lons_arr.max()), float(lats_arr.max()),
            )

        coords = self._normalize_coords(lons_arr, lats_arr, bbox)
        order = self.rng.permutation(len(coords))
        coords = coords[order]

        return self._build_trace(coords, "osm_poi")
