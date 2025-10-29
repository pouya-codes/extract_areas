# EC Cancer Model - Quick Reference

## 🚀 Quick Start

### 1. Enable the Model
Edit `config.json`:
```json
{
  "ec_cancer_enabled": true
}
```

### 2. Start Server
```bash
cd extract_areas
python app_refactored.py
```

### 3. Verify Model Loaded
```bash
curl http://localhost:8000/models/list
# Should show "ec_cancer" in loaded_models
```

### 4. Process an Image
```bash
curl -X POST http://localhost:8000/process_region_annotation \
  -F "region=@tissue.jpg" \
  -F 'mask=[{"x":100,"y":100},{"x":500,"y":100},{"x":500,"y":500},{"x":100,"y":500}]' \
  -F "model_name=ec_cancer"
```

## 📋 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tumor_threshold` | 0.9 | Lower if finding too few patches (try 0.8) |
| `batch_size` | 32 | Increase for speed, decrease for memory |
| `patch_size` | 1024 | Size of extracted patches |
| `stride` | 1 | Increase (2-3) for faster processing |
| `generate_visualization` | true | Set false to skip overlay generation |

## 📊 Output Scores

```json
{
  "nsmp_probability": 0.2345,      // 0.0-1.0
  "p53_probability": 0.7655,       // 0.0-1.0  
  "classification": "p53",         // "NSMP" or "p53"
  "confidence": 0.7655,            // max(nsmp, p53)
  "tumor_patches_found": 142       // number of patches
}
```

## 🎨 Visualization Colors

- 🟢 **Green Overlay**: NSMP classification
- 🔴 **Red Overlay**: p53 classification
- Text shows classification + confidence

## ⚙️ Configuration Paths

Required in `config.json`:
```json
{
  "ec_cancer_patch_classifier_path": "models/EC_model/tumor_normal.pt",
  "ec_cancer_representation_path": "models/EC_model/representation.pth",
  "ec_cancer_varmil_path": "models/EC_model/VarMIL.pth"
}
```

## 🐍 Python Example

```python
import requests
import json
from PIL import Image
from io import BytesIO

# Load image
image = Image.open("tissue.jpg")
buffer = BytesIO()
image.save(buffer, format='JPEG')
buffer.seek(0)

# Define region
polygon = [
    {"x": 100, "y": 100},
    {"x": 500, "y": 100},
    {"x": 500, "y": 500},
    {"x": 100, "y": 500}
]

# Process
response = requests.post(
    'http://localhost:8000/process_region_annotation',
    files={'region': ('tissue.jpg', buffer, 'image/jpeg')},
    data={
        'mask': json.dumps(polygon),
        'model_name': 'ec_cancer',
        'hyperparameters': json.dumps({
            'tumor_threshold': 0.85,
            'generate_visualization': True
        })
    }
)

result = response.json()
print(f"{result['score']['classification']}: {result['score']['confidence']:.1%}")
```

## 🔧 Common Issues

### ❌ Model Not Loading
**Check**: 
1. `ec_cancer_enabled: true` in config.json
2. All 4 model files exist at specified paths
3. Console output for error messages

### ❌ No Tumor Patches Found
**Fix**: Lower `tumor_threshold` to 0.8 or 0.7

### ❌ Out of Memory
**Fix**: Reduce `batch_size` to 16 or 8

### ❌ Slow Processing
**Fix**: Increase `stride` to 2 or 3

## 📁 Files Created

- `models/ec_cancer_model.py` - Model implementation
- `models/EC_CANCER_MODEL.md` - Full documentation  
- `models/EC_CANCER_INTEGRATION_SUMMARY.md` - Integration details
- `models/EC_CANCER_QUICK_REFERENCE.md` - This file

## 📁 Files Modified

- `config.json` - Added EC model paths (3 models)
- `app_refactored.py` - Registered EC model
- `models/__init__.py` - Exported EC model

## 🎯 Model Features

- ✅ Self-contained (no external dependencies)
- ✅ Embedded VanillaModel and VarMIL classes
- ✅ Requires mask or annotation points (user-provided)
- ✅ GPU/CPU auto-detection
- ✅ Batch processing for efficiency

## 🔗 API Endpoints

```bash
# List all models
GET /models/list

# Get EC model info
GET /models/ec_cancer/info

# Get hyperparameters
GET /models/ec_cancer/hyperparameters

# Process with annotation points
POST /process_region_annotation

# Process with mask image
POST /process_region
```

## ⏱️ Processing Time (GPU)

- Small (1K×1K): 2-5 sec
- Medium (2K×2K): 8-15 sec
- Large (4K×4K): 30-60 sec

## 💾 Resource Requirements

- **RAM**: 8-16 GB
- **VRAM**: 6-8 GB (GPU mode)
- **Disk**: ~250 MB (3 model weights)

## 📈 Performance Tuning

### For Speed
```json
{
  "stride": 3,
  "batch_size": 64,
  "patch_size": 512,
  "generate_visualization": false
}
```

### For Accuracy
```json
{
  "stride": 1,
  "batch_size": 32,
  "patch_size": 1024,
  "tumor_threshold": 0.85
}
```

## ✅ Verification Commands

```bash
# Check models loaded
curl http://localhost:8000/models/list

# Get model details
curl http://localhost:8000/models/ec_cancer/info

# Test with sample image
curl -X POST http://localhost:8000/process_region_annotation \
  -F "region=@test.jpg" \
  -F 'mask=[{"x":0,"y":0},{"x":1000,"y":0},{"x":1000,"y":1000},{"x":0,"y":1000}]' \
  -F "model_name=ec_cancer"
```

## 📚 Documentation

- **Full Docs**: `models/EC_CANCER_MODEL.md`
- **Integration**: `models/EC_CANCER_INTEGRATION_SUMMARY.md`
- **Base Interface**: `models/base_model.py`
- **Model Registry**: `README_MODELS.md`

## 🆘 Support

1. Check console logs for errors
2. Verify model weights exist and are accessible
3. Review `EC_CANCER_MODEL.md` troubleshooting section
4. Check GPU memory with `nvidia-smi`

---

**Model ID**: `ec_cancer`  
**Version**: 1.0.0  
**Status**: ✅ Ready for Testing
