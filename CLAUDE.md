# CLAUDE.md - BlindOver Server

## Project Overview

BlindOver Server is a **beverage classification API** built with FastAPI and PyTorch. It uses deep learning models to classify images of 33 different Korean beverage products. The project is designed for the BlindOver application, which helps users identify beverages through image recognition.

## Tech Stack

- **Language**: Python 3.x
- **Web Framework**: FastAPI 0.96.0 with Uvicorn ASGI server
- **Deep Learning**: PyTorch 2.0.1, TorchVision 0.15.2
- **Image Processing**: Pillow 9.5.0
- **Logging**: Loguru 0.7.0
- **Data Validation**: Pydantic 1.10.8

## Project Structure

```
blindover_server/
├── main.py                    # FastAPI application entry point
├── inference.py               # Model inference logic and image processing
├── requirements.txt           # Python dependencies
├── shufflenet_weight.pt       # Default model weights (root)
├── models/                    # Neural network architectures
│   ├── efficientnet.py        # EfficientNetV2 (20.3M params)
│   ├── mobilenet.py           # MobileNetV3 (1.62M params)
│   ├── shufflenet.py          # ShuffleNetV2 (0.44M params) - PRIMARY
│   └── mnasnet.py             # MNASNet (5.12M params)
├── weights/                   # Pre-trained model weights (~134MB total)
│   ├── shufflenet_weight.pt
│   ├── mobilenet_weight.pt
│   ├── efficientnet_weight.pt
│   ├── mnasnet_weight.pt
│   └── quantized_*.pt         # Quantized versions (smaller, faster)
├── quantization/              # Model quantization utilities
│   ├── quantization.py        # Core quantization logic
│   ├── quantized_models.py    # Quantized model wrappers
│   ├── utils.py               # Size/latency comparison tools
│   └── README.md              # Quantization documentation
└── utils/
    └── dataset.py             # Data loading and preprocessing
```

## Key Components

### API Endpoints (main.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/photo` | Upload image for beverage classification |
| `GET` | `/` | Health check endpoint |
| `POST` | `/send` | JSON data endpoint (testing) |

### Beverage Classes (33 total)

The model classifies the following Korean beverages:
- Milk products: 2%, Chocolate milk, Strawberry milk, Plain milk
- Sodas: Chilsung Cider, Coca Cola, Sprite, Pepsi, Mountain Dew, Fanta
- Zero variants: Sprite Zero, Chilsung Cider Zero, Pepsi Zero, Zero Cola
- Energy/Health: Bacchus, Vita 500, Red Bull, Pocari Sweat, Gatorade, Powerade
- Other: Milkis, McCol, Demisoda, Sol-ui-nun, Sikhye, Welch's varieties, V-Tok varieties

### Model Architecture

The primary model is **ShuffleNetV2 (x0.5)** with:
- Input: RGB images (3 channels)
- Output: 33 beverage classes
- Parameters: 0.44M (lightweight for fast inference)
- Image size: 224x224 pixels

## Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Development Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Production Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Test Image Classification (CLI)
```bash
python inference.py --model_name shufflenet --src path/to/image.jpg --weight ./weights/shufflenet_weight.pt
```

## Image Processing Pipeline

1. **Padding**: Non-square images are padded with black to maintain aspect ratio
2. **Resize**: Images are resized to 224x224 pixels
3. **Tensor Conversion**: PIL Image converted to PyTorch tensor
4. **Batch Dimension**: Unsqueeze to add batch dimension
5. **Inference**: Model outputs softmax probabilities
6. **Classification**: Returns beverage name with highest probability

## Model Quantization

Quantized models provide ~3.5-4x size reduction with minimal accuracy loss:

| Model | Original | Quantized | Reduction |
|-------|----------|-----------|-----------|
| ShuffleNetV2 | 5.5 MB | 1.5 MB | 3.58x |
| MobileNetV3 | 22.1 MB | 5.6 MB | 3.93x |

Use quantized models for deployment on resource-constrained environments.

## Code Conventions

### Language
- Code comments and variable names may be in Korean (한국어)
- API endpoint function names may use Korean characters
- Class names (beverage labels) are in Korean

### File Organization
- Model architectures in `models/` directory
- Pre-trained weights in `weights/` directory
- Utility functions in `utils/` directory
- Quantization tools in `quantization/` directory

### Import Pattern
```python
from models.shufflenet import ShuffleNetV2
from inference import inference, load_image, modelReturn
from utils.dataset import Padding, load_dataloader
```

## API Usage Example

### Image Classification Request
```bash
curl -X POST "http://localhost:8000/photo" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@beverage_image.jpg"
```

### Response
```json
"코카 콜라"
```

## Important Notes for AI Assistants

### When Modifying Code
1. The primary model used is ShuffleNetV2 - maintain compatibility with this architecture
2. All models output 33 classes - do not change `num_classes` without updating the entire pipeline
3. Image preprocessing must maintain 224x224 input size for all models
4. CORS is enabled for all origins - consider security implications for production

### When Adding Features
1. Follow FastAPI patterns for new endpoints
2. Use Loguru for logging (already configured)
3. Add new model architectures to `models/` directory following existing patterns
4. Update `inference.py:load_model()` when adding new model types

### Files to Avoid Modifying
- `weights/*.pt` - Pre-trained model weights (binary files)
- `shufflenet_weight.pt` - Root-level weight file (may be symlink or copy)

### Testing Considerations
- No formal test suite exists - consider adding pytest
- Manual testing via API endpoints
- Image files for testing should be placed in `/photo` directory (gitignored)

## Dataset Structure (for Training)

If training new models, organize data as:
```
dataset/
├── train/
│   ├── class_name_1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class_name_33/
├── valid/
└── test/
```

## Dependencies

Key dependencies from `requirements.txt`:
- `fastapi==0.96.0` - Web framework
- `uvicorn==0.22.0` - ASGI server
- `torch==2.0.1` - PyTorch
- `torchvision==0.15.2` - Computer vision utilities
- `Pillow==9.5.0` - Image processing
- `python-multipart==0.0.6` - File upload support
- `loguru==0.7.0` - Logging

## Git Workflow

- Main development on `main` branch
- Feature branches merged via pull requests
- `.gitignore` excludes: `__pycache__`, `.idea`, `*.pth`, `/photo`, `.DS_Store`

## Known Issues / TODOs

1. Model is loaded on every request in `/photo` endpoint - should be loaded once at startup
2. Commented-out code in `main.py` needs cleanup
3. No formal testing framework
4. No environment variable management for configuration
5. Some Korean function names may cause issues with certain tools
