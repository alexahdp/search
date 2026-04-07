# Geospatial Search

Geospatial search finds documents based on location: "coffee shops near me", "hotels within 5km of downtown", "all warehouses in California". This section covers the spatial data structures and algorithms that make location-based search fast and accurate.

## Table of Contents

1. [Geospatial Primitives](#geospatial-primitives)
2. [Distance Calculations](#distance-calculations)
3. [R-Trees](#r-trees)
4. [Geohashing](#geohashing)
5. [Quadtrees and Z-Order Curves](#quadtrees-and-z-order-curves)
6. [Query Types](#query-types)
7. [Combining Geo and Text Search](#combining-geo-and-text-search)
8. [Performance Optimization](#performance-optimization)
9. [References](#references)

---

## Geospatial Primitives

### Coordinate Systems

**Latitude and Longitude:**
- **Latitude**: -90° (South Pole) to +90° (North Pole)
- **Longitude**: -180° (West) to +180° (East)
- **Point**: (lat, lon) e.g., (37.7749, -122.4194) is San Francisco

**Coordinate representation:**

```
Decimal degrees: 37.7749° N, 122.4194° W
Degrees/minutes/seconds: 37°46'29.6"N 122°25'9.8"W
```

Always use decimal degrees in search systems (simpler arithmetic).

**Coordinate order confusion:**

- **GeoJSON standard**: [longitude, latitude] (x, y)
- **Common convention**: (latitude, longitude) (y, x)

Elasticsearch uses GeoJSON order: `[lon, lat]`. Be consistent!

### Shapes

**Point:** Single location `(lat, lon)`

**Bounding box (bbox):** Rectangle defined by two corners:

```
{
  "top_left": {"lat": 40.8, "lon": -74.0},
  "bottom_right": {"lat": 40.7, "lon": -73.9}
}
```

**Circle (radius search):** Center + radius

```
{
  "center": {"lat": 37.7749, "lon": -122.4194},
  "radius": "5km"
}
```

**Polygon:** Arbitrary shape defined by vertices

```
{
  "type": "polygon",
  "coordinates": [
    [
      [-122.4, 37.8],
      [-122.4, 37.7],
      [-122.5, 37.7],
      [-122.5, 37.8],
      [-122.4, 37.8]  // Close the polygon
    ]
  ]
}
```

---

## Distance Calculations

### Haversine Formula

Computes great-circle distance between two points on a sphere (assumes Earth is a sphere with radius 6,371 km).

**Formula:**

$$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

where:
- $r$ = Earth's radius (6,371 km or 3,959 miles)
- $\phi_1, \phi_2$ = latitude of point 1 and point 2 (in radians)
- $\Delta\phi = \phi_2 - \phi_1$
- $\Delta\lambda = \lambda_2 - \lambda_1$ (difference in longitude)

**Python implementation:**

```python
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    # Convert to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return R * c  # Distance in km
```

**Example:**

```python
# San Francisco to Los Angeles
haversine(37.7749, -122.4194, 34.0522, -118.2437)
# → 559.12 km
```

**Complexity:** $O(1)$ — just arithmetic.

**Accuracy:** ±0.5% (Earth is not a perfect sphere; it's an oblate spheroid).

### Vincenty Formula

More accurate distance calculation using the WGS-84 ellipsoid model (accounts for Earth's flattening).

**Complexity:** $O(1)$ but more expensive (iterative algorithm).

**Accuracy:** Accurate to within 0.5mm over distances up to 20,000 km.

**Use case:** GPS systems, surveying, aviation. Overkill for most search applications (Haversine is sufficient).

### Euclidean Distance (Flat Earth Approximation)

For small distances (<10km), treat Earth as flat:

$$d \approx \sqrt{(\Delta x)^2 + (\Delta y)^2}$$

where:
- $\Delta x = (lon_2 - lon_1) \times \cos(\text{avg\_lat}) \times 111.32$ km/degree
- $\Delta y = (lat_2 - lat_1) \times 110.574$ km/degree

**Advantage:** Much faster than Haversine (no trigonometry).

**Disadvantage:** Breaks down near poles and for long distances.

**Use case:** Fast bounding box filtering before precise distance calculation.

---

## R-Trees

R-trees are the standard spatial index for geospatial search. They organize multi-dimensional data (2D for lat/lon, can extend to 3D) for efficient range queries and nearest-neighbor search.

### Structure

An R-tree is a balanced tree where:
- **Leaf nodes** contain actual data points (or pointers to documents)
- **Internal nodes** contain **minimum bounding rectangles (MBRs)** that enclose all rectangles in child nodes

```
Example R-tree:

           [Root MBR: entire world]
          /           |            \
    [MBR: USA]   [MBR: Europe]  [MBR: Asia]
     /      \         |              |
[West US] [East US] [MBR: UK]   [MBR: China]
   |         |          |            |
 Points    Points     Points       Points
```

Each MBR is defined by:

```
MBR = {
  min_lat, max_lat,
  min_lon, max_lon
}
```

### Insertion

1. Start at root, find the child MBR that needs the least enlargement to include the new point
2. Recursively descend until reaching a leaf
3. Insert point in leaf
4. If leaf overflows (>M entries), split it and propagate split up the tree

**MBR enlargement heuristic:** Choose child that minimizes increase in MBR area.

### Range Query (Bounding Box Search)

```
RANGE_QUERY(rtree, query_bbox):
    results = []
    queue = [rtree.root]
    while queue:
        node = queue.pop()
        if node.mbr.intersects(query_bbox):
            if node.is_leaf:
                for point in node.points:
                    if query_bbox.contains(point):
                        results.append(point)
            else:
                queue.extend(node.children)
    return results
```

**Complexity:** $O(\log N + K)$ where $K$ = number of results (in practice; worst case is $O(N)$).

**Optimization:** Prune branches whose MBR doesn't intersect the query bbox.

### Radius Search (Circle Query)

```
RADIUS_SEARCH(rtree, center, radius):
    # Bounding box that encloses the circle
    bbox = bounding_box_for_circle(center, radius)

    candidates = rtree.range_query(bbox)

    # Filter candidates by precise distance
    results = []
    for point in candidates:
        if haversine(center, point) <= radius:
            results.append(point)
    return results
```

**Two-phase filtering:**
1. **Coarse filter:** Bounding box (fast, uses R-tree)
2. **Precise filter:** Haversine distance (slower, but only on candidates)

### Nearest Neighbor Search

Find the K nearest points to a query location.

**Algorithm (branch-and-bound):**

```
NEAREST_NEIGHBORS(rtree, query_point, k):
    priority_queue = []  // Min-heap by distance
    heappush(priority_queue, (0, rtree.root))
    results = []

    while priority_queue and len(results) < k:
        (dist, node) = heappop(priority_queue)

        if node.is_leaf:
            for point in node.points:
                d = haversine(query_point, point)
                results.append((d, point))
        else:
            for child in node.children:
                min_dist = min_distance(query_point, child.mbr)
                heappush(priority_queue, (min_dist, child))

    results.sort()
    return results[:k]
```

**Min distance to MBR:**

```python
def min_distance(point, mbr):
    # If point is inside MBR, distance = 0
    if mbr.contains(point):
        return 0

    # Closest point on MBR to query point
    closest_lat = clamp(point.lat, mbr.min_lat, mbr.max_lat)
    closest_lon = clamp(point.lon, mbr.min_lon, mbr.max_lon)

    return haversine(point.lat, point.lon, closest_lat, closest_lon)
```

**Complexity:** $O(\log N \cdot K)$ expected.

### R-Tree Variants

**R*-tree:** Improved split algorithm that minimizes overlap between MBRs. Better query performance than standard R-tree.

**R+ tree:** Allows MBRs to overlap but forbids points from being in multiple nodes. Can improve query performance but complicates updates.

**STR (Sort-Tile-Recursive):** Bulk-loading algorithm for static datasets. Sorts points by space-filling curve, then partitions. Much faster index construction than incremental insertion.

---

## Geohashing

Geohashing maps 2D coordinates to 1D strings, enabling spatial indexing using standard text indices.

### Geohash Algorithm

1. **Interleave bits** of latitude and longitude
2. **Base32 encode** the result

**Example:**

```
Location: San Francisco (37.7749, -122.4194)

Binary representation (simplified):
  Latitude:  37.7749  → binary bits
  Longitude: -122.4194 → binary bits

Interleave: lon[0], lat[0], lon[1], lat[1], lon[2], lat[2], ...

Base32 encode: "9q8yy"
```

**Geohash properties:**
- **Hierarchical:** Longer geohashes are more precise
- **Proximity:** Nearby locations often share prefixes (but not always!)

```
Geohash precision:

Length  Cell size
1       ±2500 km
2       ±630 km
3       ±78 km
4       ±20 km
5       ±2.4 km
6       ±610 m
7       ±76 m
8       ±19 m
9       ±2 m
10      ±60 cm
```

### Geohash for Range Queries

**Bounding box query:**

```
1. Compute geohash for bounding box corners
2. Find common prefix
3. Query all geohashes with that prefix

Example:
  Bbox corners: "9q8yy", "9q8yz"
  Common prefix: "9q8y"
  Query: geohash LIKE '9q8y%'
```

**Problem:** Edge cases. Locations near geohash boundaries may have different prefixes even if they're close.

```
San Francisco:
  Point A: (37.775, -122.419) → "9q8yy"
  Point B: (37.775, -122.420) → "9q8yw"  (different prefix!)
```

**Solution:** Query multiple geohash cells (8 neighboring cells + center cell).

### Geohash for Radius Search

```
RADIUS_SEARCH_GEOHASH(center, radius):
    # Compute geohash precision that covers the radius
    precision = geohash_precision_for_radius(radius)
    center_geohash = geohash(center, precision)

    # Get neighboring geohashes
    geohash_cells = [center_geohash] + neighbors(center_geohash)

    # Query all cells
    candidates = []
    for cell in geohash_cells:
        candidates.extend(query_geohash_prefix(cell))

    # Filter by precise distance
    results = []
    for point in candidates:
        if haversine(center, point) <= radius:
            results.append(point)

    return results
```

### Geohash in Elasticsearch

```json
{
  "mappings": {
    "properties": {
      "location": {
        "type": "geo_point"
      }
    }
  }
}

// Geohash precision query
{
  "query": {
    "geo_bounding_box": {
      "location": {
        "top_left": {"lat": 40.8, "lon": -74.0},
        "bottom_right": {"lat": 40.7, "lon": -73.9}
      }
    }
  }
}
```

Internally, Elasticsearch indexes geo_point fields using a BKD-tree (not geohashes directly), but geohashing is used for aggregations.

### Geohash Grid Aggregation

```json
{
  "aggs": {
    "location_grid": {
      "geohash_grid": {
        "field": "location",
        "precision": 5
      }
    }
  }
}

// Returns: Counts per geohash cell (heatmap data)
```

---

## Quadtrees and Z-Order Curves

### Quadtree

A tree structure that recursively subdivides 2D space into four quadrants.

```
           [Root: entire space]
          /    /      \       \
       NW     NE      SW      SE
       / \    / \     / \     / \
     ... ... ... ... ... ... ... ...
```

Each node covers a square region. Points are inserted into the appropriate quadrant.

**Properties:**
- **Adaptive resolution:** Dense areas have deeper trees
- **Simple to implement**
- **Not disk-friendly:** Pointers scattered across memory

**Use case:** In-memory spatial index (e.g., game engines, GIS systems).

### Z-Order Curve (Morton Code)

A space-filling curve that maps 2D coordinates to 1D while preserving locality.

**Algorithm:** Interleave bits of x and y coordinates (similar to geohash).

```
Point: (x=5, y=3)

Binary:
  x = 101
  y = 011

Interleave: y[0]x[0] y[1]x[1] y[2]x[2] = 010111 = 23

Z-order code: 23
```

**Visualization:**

```
Z-order traversal of a grid:

0  1  4  5
2  3  6  7
8  9  12 13
10 11 14 15
```

**Use in databases:** PostgreSQL's BRIN (Block Range Index) uses Z-order for geospatial columns.

**Elasticsearch BKD-tree** uses a variant of Z-order to index numeric/geo fields.

### BKD-Tree (Block KD-Tree)

Lucene's spatial index structure (used for `geo_point`, `geo_shape`, numeric fields).

**Key ideas:**
- **K-D tree** (k-dimensional tree): Binary space partitioning
- **Block-based:** Leaf nodes contain blocks of points (not individual points)
- **Packed structure:** Optimized for disk I/O and memory mapping

**Properties:**
- Very fast range queries: $O(\log N + K)$
- Efficient radius search
- Supports high-dimensional data (not just 2D)
- Used in Elasticsearch for all geo queries

---

## Query Types

### Bounding Box Query

"Find all points within a rectangle."

```json
// Elasticsearch
{
  "query": {
    "geo_bounding_box": {
      "location": {
        "top_left": {"lat": 40.8, "lon": -74.0},
        "bottom_right": {"lat": 40.7, "lon": -73.9}
      }
    }
  }
}
```

**Implementation:** R-tree or BKD-tree range query.

**Complexity:** $O(\log N + K)$

### Radius Query (Distance Filter)

"Find all points within X km of a location."

```json
{
  "query": {
    "geo_distance": {
      "distance": "5km",
      "location": {"lat": 37.7749, "lon": -122.4194}
    }
  }
}
```

**Implementation:**
1. Compute bounding box that encloses the circle
2. R-tree/BKD-tree range query on bbox
3. Filter candidates with Haversine distance

**Complexity:** $O(\log N + K)$ for bbox query + $O(K)$ for distance filter.

### Polygon Query

"Find all points within an arbitrary polygon."

```json
{
  "query": {
    "geo_polygon": {
      "location": {
        "points": [
          {"lat": 40.8, "lon": -74.0},
          {"lat": 40.7, "lon": -74.0},
          {"lat": 40.7, "lon": -73.9},
          {"lat": 40.8, "lon": -73.9}
        ]
      }
    }
  }
}
```

**Implementation:**
1. Compute polygon's bounding box
2. R-tree query on bbox
3. Filter candidates with point-in-polygon test

**Point-in-polygon algorithm (Ray casting):**

```python
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
```

**Complexity:** $O(V)$ where $V$ = number of polygon vertices.

### Geo Shape Queries

Elasticsearch supports arbitrary shapes (polygons, circles, linestrings, multi-polygons) via the `geo_shape` field type.

**Indexing strategy:** Recursive grid decomposition (similar to quadtree) using geohashes.

```json
{
  "mappings": {
    "properties": {
      "area": {
        "type": "geo_shape"
      }
    }
  }
}

// Query: Find docs whose geo_shape intersects a circle
{
  "query": {
    "geo_shape": {
      "area": {
        "shape": {
          "type": "circle",
          "coordinates": [-122.4194, 37.7749],
          "radius": "5km"
        },
        "relation": "intersects"
      }
    }
  }
}
```

**Relations:**
- `intersects`: Shape overlaps the query shape
- `contains`: Shape fully contains the query shape
- `within`: Shape is fully within the query shape
- `disjoint`: Shape does not touch the query shape

---

## Combining Geo and Text Search

Real-world queries often combine text and location:

```
"coffee shops within 2km of me"
"hotels in San Francisco with pool"
```

### Geo Filtering with Text Scoring

```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "name": "coffee shop" } }  // Text relevance
      ],
      "filter": [
        {
          "geo_distance": {
            "distance": "2km",
            "location": {"lat": 37.7749, "lon": -122.4194}
          }
        }
      ]
    }
  }
}
```

**Execution:**
1. Geo filter reduces candidates (10K → 50 cafes within 2km)
2. BM25 scoring on text query ("coffee shop") ranks the 50 cafes
3. Return top-10 by text relevance

### Geo Distance Scoring

Boost results by proximity:

```json
{
  "query": {
    "function_score": {
      "query": { "match": { "name": "coffee" } },
      "functions": [
        {
          "gauss": {
            "location": {
              "origin": {"lat": 37.7749, "lon": -122.4194},
              "scale": "2km",
              "decay": 0.5
            }
          }
        }
      ]
    }
  }
}
```

**Decay function:**

$$\text{score} = \text{BM25\_score} \times \exp\left(-\frac{d^2}{2 \sigma^2}\right)$$

where $d$ = distance, $\sigma$ = scale parameter.

**Effect:** Nearby results are boosted. A result 2km away gets 0.5× the score of one at the origin.

---

## Performance Optimization

### Geospatial Index Memory Usage

**BKD-tree memory (Elasticsearch):**

For 10M geo_point documents:
- Index size: ~150MB (BKD-tree structure)
- Memory-mapped, not in heap

**R-tree memory:**

In-memory R-tree for 10M points: ~500MB to 1GB (depending on implementation).

### Reducing Candidate Sets

**Problem:** Radius query with large radius (e.g., 100km) returns millions of candidates.

**Strategies:**
1. **Limit result size:** `size: 100` (return only top-100 by distance)
2. **Multi-stage filtering:** Coarse filter (bbox) → precise filter (distance) → scoring
3. **Tile-based caching:** Precompute results for common locations at fixed radii

### Geo Aggregations at Scale

**Problem:** Aggregating millions of geo points by geohash grid is expensive.

**Elasticsearch strategies:**
- **Sampling:** Aggregate on a sample of documents (faster, approximate)
- **Precision tuning:** Lower precision (larger cells) → fewer buckets → faster

```json
{
  "aggs": {
    "sample": {
      "sampler": {
        "shard_size": 1000  // Sample 1000 docs per shard
      },
      "aggs": {
        "locations": {
          "geohash_grid": {
            "field": "location",
            "precision": 4  // Coarse grid
          }
        }
      }
    }
  }
}
```

### Caching Geo Filters

Frequently used geo filters (e.g., "downtown San Francisco") can be cached as bitmaps.

**Elasticsearch query cache:** Automatically caches filter clauses, including geo_distance and geo_bounding_box.

```
Query 1: geo_distance(37.7749, -122.4194, 5km) → cache bitmap of matching docs
Query 2: Same filter → retrieve cached bitmap (near-instant)
```

**Cache invalidation:** Per-segment (when segment is merged or refreshed, cache is invalidated).

---

## References

- Guttman, A. — "R-trees: A dynamic index structure for spatial searching" (SIGMOD, 1984) — Original R-tree paper
- Beckmann, N., et al. — "The R*-tree: An efficient and robust access method for points and rectangles" (SIGMOD, 1990)
- Gaede, V. & Günther, O. — "Multidimensional access methods" (ACM Computing Surveys, 1998) — Survey of spatial indices
- Samet, H. — *Foundations of Multidimensional and Metric Data Structures* (Morgan Kaufmann, 2006) — Comprehensive textbook
- Niemeyer, G. — "Geohash" (2008) — http://geohash.org
- Malkov, Y. A. & Yashunin, D. A. — "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (TPAMI, 2018) — HNSW (alternative to R-trees for NN search)
- Lucene — "BKD-tree" source — https://github.com/apache/lucene/tree/main/lucene/core/src/java/org/apache/lucene/util/bkd
- Elasticsearch — "Geo queries" documentation — https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-queries.html
