# Photo Mosaic Creator API with Deep Zoom Support

This project provides a powerful API for creating photo mosaics with Deep Zoom functionality. It allows developers to programmatically generate intricate mosaics from a main image using a collection of smaller images, with the added capability of exploring the result in high detail through Deep Zoom.

## Features

- RESTful API for photo mosaic creation
- Deep Zoom tile generation for detailed exploration of mosaics
- Customizable mosaic parameters (grid size, color enhancement, overlay, etc.)
- Support for various color effects (default, grayscale, sepia)
- Option for full or medium resolution output
- Includes a basic web interface for testing and demonstration purposes

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/SkyAstrall/Photo-Mosaic-Creator-with-Deep-Zoom-API.git
   cd Photo-Mosaic-Creator-with-Deep-Zoom-API
   ```

2. Set up a virtual environment (recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Requirements

The project depends on the following Python packages:

```
click==8.1.7
Flask==2.0.3
itsdangerous==2.2.0
Jinja2==3.1.4
joblib==1.4.2
MarkupSafe==2.1.5
numpy==2.1.1
Pillow==8.4.0
threadpoolctl==3.5.0
Werkzeug==2.0.3
```

## API Documentation

### Create Mosaic

Creates a photo mosaic and generates Deep Zoom tiles.

**Endpoint:** `/`

**Method:** POST

**Content-Type:** multipart/form-data

**Parameters:**

- `main_photo` (file, required): The main image file for the mosaic
- `collection_folder` (string, required): Path to the folder with smaller images
- `size` (integer, optional): Grid size for the mosaic (default: 20)
- `distance` (integer, optional): Minimum cells between repeated images (default: 10)
- `color_enhance` (float, optional): Color enhancement factor, 0-1 (default: 0.5)
- `source_overlay` (float, optional): Source image overlay factor, 0-1 (default: 0.3)
- `color_effect` (string, optional): Color effect to apply ('default', 'grayscale', or 'sepia')
- `render_type` (string, optional): Rendering type ('full' or 'medium')

**Success Response:**

```json
{
  "mosaic_url": "/static/mosaic.jpg",
  "deep_zoom_url": "/static/deep_zoom/dzc_output.dzi"
}
```

**Error Response:**

```json
{
  "error": "Error message"
}
```

### Serve Deep Zoom Tiles

Serves individual Deep Zoom tiles.

**Endpoint:** `/static/deep_zoom/<path:filename>`

**Method:** GET

**Response:** The requested Deep Zoom tile image

## Usage Examples

### Using cURL

```bash
curl -X POST -F "main_photo=@path/to/main_image.jpg" \
     -F "collection_folder=/path/to/image/collection" \
     -F "size=30" \
     -F "color_effect=sepia" \
     http://localhost:5000/
```

### Using Python requests

```python
import requests

url = "http://localhost:5000/"
files = {"main_photo": open("path/to/main_image.jpg", "rb")}
data = {
    "collection_folder": "/path/to/image/collection",
    "size": 30,
    "color_effect": "sepia"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Web Interface

A basic web interface is included for testing and demonstration. To use it:

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`
3. Use the form to create mosaics and test the Deep Zoom functionality

## Project Structure

- `app.py`: Main application file with API endpoints and mosaic creation logic
- `templates/index.html`: HTML template for the testing interface
- `static/`: Directory for generated mosaics and Deep Zoom tiles
- `requirements.txt`: List of Python dependencies

## Acknowledgements

- Flask: Web framework for the API
- NumPy: Numerical operations for image processing
- Pillow: Core image processing functionality
- OpenSeadragon: Deep Zoom viewer implementation
