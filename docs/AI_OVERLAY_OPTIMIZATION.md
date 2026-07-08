# AI Overlay Optimization - Development Reference

## Overview

This document outlines the plan to optimize AI model result visualization in the pathology viewer platform. The current approach of rendering results as a single large PNG overlay is inefficient for WSI (Whole Slide Images). We propose two complementary solutions:

1. **Tile-Based Overlay** - For heatmaps and activation maps
2. **Vector-Based Overlay** - For HoVer-Net nucleus segmentation results

---

## Current Architecture (Problem)

```
┌──────────────┐     ┌──────────────┐     ┌────────────────────┐
│  AI Model    │────▶│  Single PNG  │────▶│  Browser Overlay   │
│  Processing  │     │  (50-500MB)  │     │  (Memory Heavy)    │
└──────────────┘     └──────────────┘     └────────────────────┘

Problems:
- Slow download (10-60 seconds)
- High memory usage
- Browser may crash on large images
- All-or-nothing loading
- No interactivity
- Pixelated on zoom
```

---

## JIRA Ticket #1: Tile-Based Overlay for Heatmaps

### Summary
Implement tile pyramid overlay system for AI heatmaps and activation maps.

### Priority: High | Story Points: 13

### Description
Generate overlay tiles on-demand at multiple zoom levels, similar to how WSI images are rendered. Only visible tiles at the current zoom level are loaded.

### Technical Implementation

#### Backend API Endpoint

```python
# File: routes/tiles.py

from fastapi import APIRouter, Response
from services.tile_generator import TileGenerator
from services.tile_cache import TileCache

router = APIRouter()
tile_cache = TileCache()

@router.get("/overlay_tiles/{image_id}/{region_id}/{model_name}/{z}/{x}/{y}.png")
async def get_overlay_tile(
    image_id: str,
    region_id: str,
    model_name: str,
    z: int,  # zoom level
    x: int,  # tile x coordinate
    y: int   # tile y coordinate
):
    """
    Serve overlay tiles for AI results.
    
    Tile pyramid structure:
    - Level 0: 1 tile covers entire region
    - Level 1: 4 tiles (2x2)
    - Level 2: 16 tiles (4x4)
    - ... up to full resolution
    """
    cache_key = f"{image_id}_{region_id}_{model_name}_{z}_{x}_{y}"
    
    # Check cache first
    cached_tile = tile_cache.get(cache_key)
    if cached_tile:
        return Response(content=cached_tile, media_type="image/png")
    
    # Generate tile on-demand
    tile_generator = TileGenerator(image_id, region_id, model_name)
    tile_png = tile_generator.generate_tile(z, x, y)
    
    # Cache the tile
    tile_cache.set(cache_key, tile_png, ttl=3600)  # 1 hour TTL
    
    return Response(content=tile_png, media_type="image/png")


@router.delete("/overlay_tiles/{image_id}/{region_id}/{model_name}")
async def invalidate_tiles(image_id: str, region_id: str, model_name: str):
    """Invalidate cached tiles when new AI results are generated."""
    pattern = f"{image_id}_{region_id}_{model_name}_*"
    tile_cache.delete_pattern(pattern)
    return {"status": "ok", "message": "Tiles invalidated"}
```

#### Tile Generator Service

```python
# File: services/tile_generator.py

import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

class TileGenerator:
    """Generate tile pyramid from AI heatmap results."""
    
    TILE_SIZE = 256  # Standard tile size
    
    def __init__(self, image_id: str, region_id: str, model_name: str):
        self.image_id = image_id
        self.region_id = region_id
        self.model_name = model_name
        self.heatmap = self._load_heatmap()
    
    def _load_heatmap(self) -> np.ndarray:
        """Load the full-resolution heatmap from storage."""
        # Load from disk/database based on IDs
        heatmap_path = Path(f"results/{self.image_id}/{self.region_id}/{self.model_name}_heatmap.npy")
        if heatmap_path.exists():
            return np.load(heatmap_path)
        raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")
    
    def get_max_zoom(self) -> int:
        """Calculate maximum zoom level based on heatmap size."""
        h, w = self.heatmap.shape[:2]
        max_dim = max(h, w)
        return int(np.ceil(np.log2(max_dim / self.TILE_SIZE)))
    
    def generate_tile(self, z: int, x: int, y: int) -> bytes:
        """
        Generate a single tile at given zoom level and coordinates.
        
        Args:
            z: Zoom level (0 = zoomed out, max = full resolution)
            x: Tile x coordinate
            y: Tile y coordinate
            
        Returns:
            PNG image bytes
        """
        max_zoom = self.get_max_zoom()
        
        if z > max_zoom:
            return self._empty_tile()
        
        # Calculate scale factor for this zoom level
        scale = 2 ** (max_zoom - z)
        
        # Calculate source region in full-resolution heatmap
        src_size = self.TILE_SIZE * scale
        src_x = x * src_size
        src_y = y * src_size
        
        h, w = self.heatmap.shape[:2]
        
        # Check bounds
        if src_x >= w or src_y >= h:
            return self._empty_tile()
        
        # Extract region from heatmap
        src_x_end = min(src_x + src_size, w)
        src_y_end = min(src_y + src_size, h)
        
        region = self.heatmap[src_y:src_y_end, src_x:src_x_end]
        
        # Resize to tile size
        if region.shape[0] == 0 or region.shape[1] == 0:
            return self._empty_tile()
        
        # Convert to PIL Image and resize
        if region.ndim == 2:
            # Grayscale heatmap - apply colormap
            region_colored = self._apply_colormap(region)
        else:
            region_colored = region
        
        img = Image.fromarray(region_colored.astype(np.uint8))
        img = img.resize((self.TILE_SIZE, self.TILE_SIZE), Image.LANCZOS)
        
        # Convert to PNG bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        return buffer.getvalue()
    
    def _apply_colormap(self, grayscale: np.ndarray) -> np.ndarray:
        """Apply a colormap to grayscale heatmap."""
        import cv2
        
        # Normalize to 0-255
        normalized = ((grayscale - grayscale.min()) / 
                      (grayscale.max() - grayscale.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap (e.g., JET, VIRIDIS)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Add alpha channel (transparent where value is low)
        alpha = (normalized > 20).astype(np.uint8) * 200  # Threshold for transparency
        
        rgba = np.dstack([
            colored[:, :, 2],  # R (OpenCV is BGR)
            colored[:, :, 1],  # G
            colored[:, :, 0],  # B
            alpha               # A
        ])
        
        return rgba
    
    def _empty_tile(self) -> bytes:
        """Generate an empty transparent tile."""
        img = Image.new('RGBA', (self.TILE_SIZE, self.TILE_SIZE), (0, 0, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
```

#### Tile Cache Service

```python
# File: services/tile_cache.py

import redis
from pathlib import Path
from typing import Optional
import hashlib

class TileCache:
    """
    Cache for overlay tiles.
    
    Supports:
    - Redis (fast, in-memory, TTL support)
    - Disk (persistent, larger capacity)
    - Hybrid (Redis for hot tiles, disk for cold)
    """
    
    def __init__(self, 
                 use_redis: bool = True,
                 redis_url: str = "redis://localhost:6379",
                 disk_path: str = "./tile_cache"):
        
        self.use_redis = use_redis
        self.disk_path = Path(disk_path)
        self.disk_path.mkdir(parents=True, exist_ok=True)
        
        if use_redis:
            try:
                self.redis = redis.from_url(redis_url)
                self.redis.ping()
            except redis.ConnectionError:
                print("Warning: Redis not available, using disk cache only")
                self.use_redis = False
                self.redis = None
    
    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.disk_path / f"{key_hash}.png"
    
    def get(self, key: str) -> Optional[bytes]:
        """Get tile from cache."""
        # Try Redis first
        if self.use_redis:
            data = self.redis.get(key)
            if data:
                return data
        
        # Try disk
        path = self._key_to_path(key)
        if path.exists():
            return path.read_bytes()
        
        return None
    
    def set(self, key: str, data: bytes, ttl: int = 3600) -> None:
        """Store tile in cache."""
        # Store in Redis with TTL
        if self.use_redis:
            self.redis.setex(key, ttl, data)
        
        # Also store on disk (persistent)
        path = self._key_to_path(key)
        path.write_bytes(data)
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        count = 0
        
        if self.use_redis:
            keys = self.redis.keys(pattern)
            if keys:
                count = self.redis.delete(*keys)
        
        # For disk, we'd need to track key->path mapping
        # Simplified: just clear the entire cache for this region
        
        return count
```

#### Frontend: OpenLayers TileLayer

```javascript
// File: frontend/annotator/src/viewer/layers/HeatmapTileLayer.js

import TileLayer from 'ol/layer/Tile';
import XYZ from 'ol/source/XYZ';

/**
 * Create a tile layer for AI heatmap overlay.
 * 
 * @param {Object} options
 * @param {string} options.imageId - WSI image ID
 * @param {string} options.regionId - Processed region ID
 * @param {string} options.modelName - AI model name
 * @param {number} options.opacity - Layer opacity (0-1)
 * @param {number} options.maxZoom - Maximum zoom level
 * @returns {TileLayer}
 */
export function createHeatmapTileLayer({
    imageId,
    regionId,
    modelName,
    opacity = 0.6,
    maxZoom = 18
}) {
    const baseUrl = process.env.VUE_APP_API_URL || '';
    
    const source = new XYZ({
        url: `${baseUrl}/overlay_tiles/${imageId}/${regionId}/${modelName}/{z}/{x}/{y}.png`,
        tileSize: 256,
        maxZoom: maxZoom,
        crossOrigin: 'anonymous',
        
        // Handle tile load errors gracefully
        tileLoadFunction: (tile, src) => {
            const image = tile.getImage();
            image.onerror = () => {
                // Set transparent image on error
                image.src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
            };
            image.src = src;
        }
    });
    
    const layer = new TileLayer({
        source: source,
        opacity: opacity,
        visible: true,
        zIndex: 10,  // Above base WSI layer
        properties: {
            name: 'heatmap-overlay',
            modelName: modelName,
            regionId: regionId
        }
    });
    
    return layer;
}

/**
 * Refresh heatmap tiles after new AI results.
 * 
 * @param {TileLayer} layer - The heatmap tile layer
 */
export function refreshHeatmapLayer(layer) {
    const source = layer.getSource();
    
    // Add timestamp to bust cache
    const currentUrl = source.getUrls()[0];
    const newUrl = currentUrl.includes('?') 
        ? currentUrl.replace(/\?.*/, `?t=${Date.now()}`)
        : `${currentUrl}?t=${Date.now()}`;
    
    source.setUrl(newUrl);
    source.refresh();
}
```

### Acceptance Criteria

- [ ] Backend generates overlay tiles on-demand at multiple zoom levels
- [ ] Tiles are cached (Redis + disk) to avoid regeneration
- [ ] Frontend displays overlay as TileLayer in OpenLayers
- [ ] Overlay tiles load progressively as user pans/zooms
- [ ] Memory usage stays constant regardless of image size
- [ ] Overlay opacity can be adjusted via UI control
- [ ] Cache invalidation when new AI results are generated
- [ ] Loading indicators shown for pending tiles

---

## JIRA Ticket #2: Vector-Based Overlay for HoVer-Net

### Summary
Implement vector-based rendering for HoVer-Net nucleus segmentation results.

### Priority: High | Story Points: 21

### Description
Return nucleus data (contours, centroids, types) as GeoJSON instead of rasterized PNG. This enables efficient rendering, infinite zoom without pixelation, dynamic styling, and interactive filtering.

### Technical Implementation

#### Modified HoVer-Net Output

```python
# File: models/hovernet_model.py (modifications)

from typing import Dict, Any, List, Tuple
import numpy as np
from shapely.geometry import Polygon
from shapely.simplify import simplify

class HoVerNetModel(BaseAIModel):
    
    def process(
        self,
        image: Image.Image,
        mask: Optional[Image.Image] = None,
        annotation_points: Optional[List[Tuple[float, float]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process image with HoVer-Net.
        
        NEW: Returns vector data in addition to (or instead of) overlay image.
        """
        # ... existing processing code ...
        
        # Post-process predictions
        pred_inst, inst_info_dict = self._post_process(pred_map)
        
        # NEW: Generate vector output
        vector_data = self._generate_vector_output(
            inst_info_dict,
            simplify_tolerance=hyperparameters.get('simplify_tolerance', 1.0)
        )
        
        # Compile scores
        scores = self._compute_scores(inst_info_dict)
        str_result = self._format_result_string(scores)
        
        return {
            'success': True,
            'vector_data': vector_data,  # NEW: GeoJSON data
            'scores': scores,
            'str_result': str_result,
            'metadata': {
                'model_mode': self.model_mode,
                'nr_types': self.nr_types,
                'total_nuclei': scores.get('total_nuclei', 0)
            }
        }
    
    def _generate_vector_output(
        self,
        inst_info_dict: Dict,
        simplify_tolerance: float = 1.0
    ) -> Dict[str, Any]:
        """
        Convert nucleus instances to GeoJSON format.
        
        Args:
            inst_info_dict: Dictionary of nucleus information
            simplify_tolerance: Douglas-Peucker simplification tolerance
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for inst_id, info in inst_info_dict.items():
            # Get contour points
            contour = info['contour']
            
            if len(contour) < 3:
                continue
            
            # Simplify contour using Douglas-Peucker algorithm
            try:
                polygon = Polygon(contour)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Fix invalid polygons
                
                simplified = polygon.simplify(
                    simplify_tolerance, 
                    preserve_topology=True
                )
                
                # Convert back to coordinate list
                coords = list(simplified.exterior.coords)
            except Exception:
                coords = contour.tolist() if isinstance(contour, np.ndarray) else contour
            
            # Get nucleus type info
            type_id = info.get('type')
            type_name = "Unknown"
            type_color = [128, 128, 128]  # Gray default
            
            if type_id is not None and self.type_info_dict:
                type_info = self.type_info_dict.get(type_id)
                if type_info:
                    type_name = type_info[0]
                    type_color = list(type_info[1])
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "id": int(inst_id),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "id": int(inst_id),
                    "type_id": type_id,
                    "type_name": type_name,
                    "type_color": type_color,
                    "centroid": info['centroid'].tolist(),
                    "area": float(info['area']),
                    "bbox": info['bbox'].tolist()
                }
            }
            
            features.append(feature)
        
        # Create FeatureCollection with summary
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "total_nuclei": len(features),
                "by_type": self._count_by_type(features),
                "type_colors": self._get_type_colors()
            }
        }
    
    def _count_by_type(self, features: List[Dict]) -> Dict[str, int]:
        """Count nuclei by type."""
        counts = {}
        for f in features:
            type_name = f['properties']['type_name']
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def _get_type_colors(self) -> Dict[str, List[int]]:
        """Get color mapping for each type."""
        if not self.type_info_dict:
            return {}
        return {
            info[0]: list(info[1]) 
            for info in self.type_info_dict.values()
        }
```

#### Backend API Endpoint

```python
# File: app_refactored.py (new endpoint)

@app.post("/process_region_nuclei")
async def process_region_nuclei(
    region: UploadFile = File(...),
    mask: str = Form(...),
    region_id: str = Form(""),
    hyperparameters: str = Form(None)
):
    """
    Process region with HoVer-Net and return vector data.
    
    Returns GeoJSON with nucleus contours, centroids, and types.
    """
    model = model_registry.get_model("hovernet")
    if not model:
        return JSONResponse(
            {"status": "error", "message": "HoVer-Net model not loaded"},
            status_code=400
        )
    
    # Parse inputs
    region_bytes = await region.read()
    region_image = Image.open(BytesIO(region_bytes)).convert("RGB")
    
    points_payload = json.loads(mask)
    annotation_points = parse_points(points_payload)
    
    # Create mask from polygon
    mask_image = create_mask_from_polygon(region_image.size, annotation_points)
    
    # Parse hyperparameters
    params = json.loads(hyperparameters) if hyperparameters else {}
    params['output_format'] = 'vector'  # Request vector output
    
    # Process with HoVer-Net
    result = model.process(
        image=region_image,
        mask=mask_image,
        hyperparameters=params
    )
    
    if not result.get('success'):
        return JSONResponse(
            {"status": "error", "message": result.get('error', 'Processing failed')},
            status_code=500
        )
    
    return JSONResponse({
        "status": "success",
        "region_id": region_id,
        "geojson": result['vector_data'],
        "summary": result['scores'],
        "str_result": result['str_result']
    })
```

#### Frontend: Vector Layer

```javascript
// File: frontend/annotator/src/viewer/layers/NucleusVectorLayer.js

import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import GeoJSON from 'ol/format/GeoJSON';
import { Style, Fill, Stroke, Circle } from 'ol/style';
import { Cluster } from 'ol/source';

/**
 * Create a vector layer for nucleus contours.
 */
export function createNucleusVectorLayer() {
    const source = new VectorSource();
    
    const layer = new VectorLayer({
        source: source,
        style: nucleusStyleFunction,
        zIndex: 20,
        properties: {
            name: 'nucleus-contours'
        }
    });
    
    return layer;
}

/**
 * Style function for nucleus features.
 */
function nucleusStyleFunction(feature, resolution) {
    const props = feature.getProperties();
    const typeColor = props.type_color || [128, 128, 128];
    const [r, g, b] = typeColor;
    
    // Adjust style based on zoom level
    const strokeWidth = resolution < 0.5 ? 2 : 1;
    const fillOpacity = resolution < 1 ? 0.4 : 0.2;
    
    return new Style({
        fill: new Fill({
            color: `rgba(${r}, ${g}, ${b}, ${fillOpacity})`
        }),
        stroke: new Stroke({
            color: `rgb(${r}, ${g}, ${b})`,
            width: strokeWidth
        })
    });
}

/**
 * Load nuclei from GeoJSON data.
 * 
 * @param {VectorLayer} layer - The nucleus vector layer
 * @param {Object} geojsonData - GeoJSON FeatureCollection
 * @param {Array} regionOffset - [x, y] offset for region coordinates
 */
export function loadNuclei(layer, geojsonData, regionOffset = [0, 0]) {
    const source = layer.getSource();
    source.clear();
    
    const format = new GeoJSON();
    const features = format.readFeatures(geojsonData, {
        featureProjection: 'EPSG:3857'  // Or your projection
    });
    
    // Apply region offset to coordinates
    if (regionOffset[0] !== 0 || regionOffset[1] !== 0) {
        features.forEach(feature => {
            const geometry = feature.getGeometry();
            geometry.translate(regionOffset[0], regionOffset[1]);
        });
    }
    
    source.addFeatures(features);
    
    return features.length;
}

/**
 * Filter visible nucleus types.
 * 
 * @param {VectorLayer} layer - The nucleus vector layer
 * @param {Array<string>} visibleTypes - Array of type names to show
 */
export function filterNucleusByType(layer, visibleTypes) {
    layer.setStyle((feature, resolution) => {
        const typeName = feature.get('type_name');
        
        if (!visibleTypes.includes(typeName)) {
            return null;  // Hide feature
        }
        
        return nucleusStyleFunction(feature, resolution);
    });
}
```

#### Frontend: WebGL Centroids Layer (High Performance)

```javascript
// File: frontend/annotator/src/viewer/layers/NucleusCentroidLayer.js

import WebGLPointsLayer from 'ol/layer/WebGLPoints';
import VectorSource from 'ol/source/Vector';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';

/**
 * Create a WebGL-accelerated layer for nucleus centroids.
 * Can handle 100k+ points efficiently.
 */
export function createCentroidLayer(typeColors) {
    const source = new VectorSource();
    
    // Build color match expression for WebGL
    const colorMatch = buildColorMatchExpression(typeColors);
    
    const layer = new WebGLPointsLayer({
        source: source,
        style: {
            symbol: {
                symbolType: 'circle',
                size: [
                    'interpolate',
                    ['linear'],
                    ['zoom'],
                    0, 2,    // At zoom 0: 2px
                    5, 4,    // At zoom 5: 4px
                    10, 8,   // At zoom 10: 8px
                    15, 12   // At zoom 15: 12px
                ],
                color: colorMatch,
                opacity: 0.8
            }
        },
        zIndex: 25,
        properties: {
            name: 'nucleus-centroids'
        }
    });
    
    return layer;
}

/**
 * Build WebGL color match expression from type colors.
 */
function buildColorMatchExpression(typeColors) {
    // Default: ['match', ['get', 'type_id'], ...cases, defaultColor]
    const cases = [];
    
    Object.entries(typeColors).forEach(([typeName, color]) => {
        // Find type_id for this type (you may need to pass this mapping)
        // For now, use type_name as key
        cases.push(typeName);
        cases.push(`rgb(${color[0]}, ${color[1]}, ${color[2]})`);
    });
    
    return [
        'match',
        ['get', 'type_name'],
        ...cases,
        'rgb(128, 128, 128)'  // Default gray
    ];
}

/**
 * Load centroids from GeoJSON nuclei data.
 * 
 * @param {WebGLPointsLayer} layer - The centroid layer
 * @param {Object} geojsonData - GeoJSON FeatureCollection with nuclei
 * @param {Array} regionOffset - [x, y] offset
 */
export function loadCentroids(layer, geojsonData, regionOffset = [0, 0]) {
    const source = layer.getSource();
    source.clear();
    
    const features = geojsonData.features.map(f => {
        const [cx, cy] = f.properties.centroid;
        const point = new Point([
            cx + regionOffset[0],
            cy + regionOffset[1]
        ]);
        
        return new Feature({
            geometry: point,
            id: f.id,
            type_id: f.properties.type_id,
            type_name: f.properties.type_name,
            area: f.properties.area
        });
    });
    
    source.addFeatures(features);
    
    return features.length;
}
```

#### Frontend: Nucleus Legend Component

```vue
<!-- File: frontend/annotator/src/components/NucleusLegend.vue -->

<template>
  <div class="nucleus-legend" v-if="visible">
    <div class="legend-header">
      <h4>Nucleus Types</h4>
      <span class="total-count">{{ totalCount }} nuclei</span>
    </div>
    
    <div class="legend-items">
      <div 
        v-for="(item, index) in legendItems" 
        :key="item.name"
        class="legend-item"
        :class="{ disabled: !item.visible }"
        @click="toggleType(item.name)"
      >
        <span 
          class="color-box" 
          :style="{ backgroundColor: item.colorHex }"
        ></span>
        <span class="type-name">{{ item.name }}</span>
        <span class="type-count">({{ item.count }})</span>
        <input 
          type="checkbox" 
          :checked="item.visible"
          @click.stop="toggleType(item.name)"
        />
      </div>
    </div>
    
    <div class="legend-actions">
      <button @click="showAll">Show All</button>
      <button @click="hideAll">Hide All</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NucleusLegend',
  
  props: {
    nucleiData: {
      type: Object,
      default: null
    },
    visible: {
      type: Boolean,
      default: true
    }
  },
  
  data() {
    return {
      visibleTypes: []
    };
  },
  
  computed: {
    totalCount() {
      return this.nucleiData?.properties?.total_nuclei || 0;
    },
    
    legendItems() {
      if (!this.nucleiData) return [];
      
      const byType = this.nucleiData.properties?.by_type || {};
      const typeColors = this.nucleiData.properties?.type_colors || {};
      
      return Object.entries(byType).map(([name, count]) => {
        const color = typeColors[name] || [128, 128, 128];
        return {
          name,
          count,
          color,
          colorHex: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
          visible: this.visibleTypes.includes(name)
        };
      }).sort((a, b) => b.count - a.count);
    }
  },
  
  watch: {
    nucleiData: {
      immediate: true,
      handler(data) {
        if (data) {
          // Initially show all types
          this.visibleTypes = Object.keys(data.properties?.by_type || {});
        }
      }
    }
  },
  
  methods: {
    toggleType(typeName) {
      const index = this.visibleTypes.indexOf(typeName);
      if (index === -1) {
        this.visibleTypes.push(typeName);
      } else {
        this.visibleTypes.splice(index, 1);
      }
      this.$emit('filter-changed', this.visibleTypes);
    },
    
    showAll() {
      this.visibleTypes = this.legendItems.map(item => item.name);
      this.$emit('filter-changed', this.visibleTypes);
    },
    
    hideAll() {
      this.visibleTypes = [];
      this.$emit('filter-changed', this.visibleTypes);
    }
  }
};
</script>

<style scoped>
.nucleus-legend {
  position: absolute;
  bottom: 20px;
  right: 20px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
  min-width: 200px;
  z-index: 1000;
}

.legend-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #eee;
}

.legend-header h4 {
  margin: 0;
  font-size: 14px;
}

.total-count {
  font-size: 12px;
  color: #666;
}

.legend-item {
  display: flex;
  align-items: center;
  padding: 4px 0;
  cursor: pointer;
  transition: opacity 0.2s;
}

.legend-item:hover {
  background: #f5f5f5;
}

.legend-item.disabled {
  opacity: 0.5;
}

.color-box {
  width: 16px;
  height: 16px;
  border-radius: 3px;
  margin-right: 8px;
  border: 1px solid rgba(0, 0, 0, 0.2);
}

.type-name {
  flex: 1;
  font-size: 13px;
}

.type-count {
  font-size: 12px;
  color: #888;
  margin-right: 8px;
}

.legend-actions {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #eee;
  display: flex;
  gap: 8px;
}

.legend-actions button {
  flex: 1;
  padding: 4px 8px;
  font-size: 12px;
  border: 1px solid #ddd;
  background: #f9f9f9;
  border-radius: 4px;
  cursor: pointer;
}

.legend-actions button:hover {
  background: #eee;
}
</style>
```

### Acceptance Criteria

- [ ] Backend returns nucleus data as JSON (contours, centroids, types, areas)
- [ ] Contours are simplified using Douglas-Peucker algorithm
- [ ] Frontend renders nuclei as vector features in OpenLayers
- [ ] Nuclei are color-coded by type with legend
- [ ] User can toggle visibility of nucleus types
- [ ] Click on nucleus shows details (type, area, etc.)
- [ ] Performance is acceptable with 50k+ nuclei
- [ ] Clustering applied at low zoom levels (optional)

---

## File Structure

```
extract_areas/
├── models/
│   ├── hovernet_model.py          # MODIFIED: Add vector output
│   └── base_model.py
├── services/                       # NEW DIRECTORY
│   ├── __init__.py
│   ├── tile_generator.py          # NEW: Tile pyramid generation
│   ├── tile_cache.py              # NEW: Redis/disk caching
│   └── contour_simplifier.py      # NEW: Douglas-Peucker
├── routes/                         # NEW DIRECTORY
│   ├── __init__.py
│   └── tiles.py                   # NEW: Tile endpoints
├── app_refactored.py              # MODIFIED: New endpoints
└── docs/
    └── AI_OVERLAY_OPTIMIZATION.md # THIS FILE

frontend/annotator/src/
├── components/
│   ├── NucleusLegend.vue          # NEW
│   ├── LayerControls.vue          # NEW
│   └── NucleusDetails.vue         # NEW
├── viewer/
│   ├── layers/                    # NEW DIRECTORY
│   │   ├── HeatmapTileLayer.js    # NEW
│   │   ├── NucleusVectorLayer.js  # NEW
│   │   └── NucleusCentroidLayer.js # NEW
│   └── Viewer.js                  # MODIFIED
└── store/
    └── modules/
        └── nuclei.js              # NEW: Vuex module
```

---

## Performance Expectations

| Metric | Current (PNG) | Tile-Based | Vector-Based |
|--------|---------------|------------|--------------|
| Initial Load | 50-500 MB | ~256 KB | 1-5 MB |
| Memory Usage | High | Low | Medium |
| Zoom Quality | Pixelated | Pixelated | Sharp |
| Interactivity | None | None | Full |
| Load Time | 10-60 sec | <1 sec | 2-5 sec |

---

## Dependencies

### Backend
- `shapely` - For polygon simplification (Douglas-Peucker)
- `redis` - For tile caching (optional)

```bash
pip install shapely redis
```

### Frontend
- OpenLayers 7+ (already installed)
- WebGL support in browser (for WebGLPointsLayer)

---

## References

- [OpenLayers TileLayer Documentation](https://openlayers.org/en/latest/apidoc/module-ol_layer_Tile-TileLayer.html)
- [OpenLayers WebGLPointsLayer](https://openlayers.org/en/latest/apidoc/module-ol_layer_WebGLPoints-WebGLPointsLayer.html)
- [Douglas-Peucker Algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
- [GeoJSON Specification](https://geojson.org/)
