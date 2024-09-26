from flask import Flask, request, render_template, send_file, jsonify
import numpy as np
from PIL import Image
import os
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import math
import random

app = Flask(__name__)


def get_average_color(image):
    img_array = np.array(image, dtype=np.float32)
    return tuple(int(x) for x in img_array.mean(axis=(0, 1)))


def apply_color_effect(image, effect):
    if effect == "grayscale":
        return image.convert("L").convert("RGB")
    elif effect == "sepia":
        sepia_matrix = [
            0.393,
            0.769,
            0.189,
            0,
            0.349,
            0.686,
            0.168,
            0,
            0.272,
            0.534,
            0.131,
            0,
        ]
        return image.convert("RGB", matrix=sepia_matrix)
    return image


def create_photo_mosaic(
    main_photo,
    collection_folder,
    size,
    distance,
    color_enhance,
    source_overlay,
    color_effect,
    render_type,
):
    try:
        main_image = Image.open(main_photo).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error opening main image: {str(e)}")

    original_width, original_height = main_image.size
    cell_width, cell_height = max(1, original_width // size), max(
        1, original_height // size
    )

    collection_images = []
    for filename in os.listdir(collection_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(collection_folder, filename)).convert(
                    "RGB"
                )
                img = img.resize((cell_width, cell_height), Image.LANCZOS)
                collection_images.append(img)
            except Exception as e:
                print(f"Error processing image {filename}: {str(e)}")

    if not collection_images:
        raise ValueError("No valid images found in the collection folder")

    mosaic = Image.new("RGB", (original_width, original_height))
    main_array = np.array(main_image, dtype=np.uint8)

    total_cells = size * size
    image_count = len(collection_images)

    # Create a list of indices that will be used to select images
    image_indices = list(range(image_count))
    random.shuffle(image_indices)
    image_index = 0

    def process_cell(index):
        nonlocal image_index
        x = (index % size) * cell_width
        y = (index // size) * cell_height
        cell = main_array[y : y + cell_height, x : x + cell_width]
        avg_color = tuple(int(x) for x in cell.mean(axis=(0, 1)))

        # Select the next image and move to the next index
        best_match = collection_images[image_indices[image_index]]
        image_index = (image_index + 1) % image_count

        if color_enhance > 0:
            enhanced = Image.blend(
                best_match, Image.new("RGB", best_match.size, avg_color), color_enhance
            )
        else:
            enhanced = best_match

        return (x, y, enhanced)

    with ThreadPoolExecutor() as executor:
        future_to_cell = {
            executor.submit(process_cell, i): i for i in range(total_cells)
        }

        for future in as_completed(future_to_cell):
            x, y, cell_image = future.result()
            mosaic.paste(cell_image, (x, y))

    if source_overlay > 0:
        mosaic = Image.blend(mosaic, main_image, source_overlay)

    mosaic = apply_color_effect(mosaic, color_effect)

    if render_type == "medium":
        mosaic = mosaic.resize(
            (original_width // 2, original_height // 2), Image.LANCZOS
        )
        # Add watermark here if needed

    return mosaic


def create_deep_zoom_tiles(image, output_dir, tile_size=256, overlap=1):
    width, height = image.size
    max_level = math.ceil(math.log(max(width, height), 2))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for level in range(max_level + 1):
        level_dir = os.path.join(output_dir, f"dzc_output_files/{str(level)}")
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)

        scale = 2 ** (max_level - level)
        level_width = max(1, width // scale)
        level_height = max(1, height // scale)

        level_image = image.resize((level_width, level_height), Image.LANCZOS)

        columns = math.ceil(level_width / tile_size)
        rows = math.ceil(level_height / tile_size)

        for row in range(rows):
            for col in range(columns):
                x = col * tile_size
                y = row * tile_size
                tile_width = min(tile_size + overlap, level_width - x)
                tile_height = min(tile_size + overlap, level_height - y)

                if tile_width <= 0 or tile_height <= 0:
                    continue

                tile = level_image.crop((x, y, x + tile_width, y + tile_height))
                tile_filename = f"{col}_{row}.jpg"
                tile.save(os.path.join(level_dir, tile_filename), "JPEG", quality=85)

    dzi_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="{overlap}"
  TileSize="{tile_size}">
  <Size Width="{width}" Height="{height}"/>
</Image>"""

    with open(os.path.join(output_dir, "dzc_output.dzi"), "w") as f:
        f.write(dzi_content)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "main_photo" not in request.files:
                return jsonify({"error": "No file part"}), 400
            main_photo = request.files["main_photo"]
            if main_photo.filename == "":
                return jsonify({"error": "No selected file"}), 400

            collection_folder = request.form["collection_folder"]
            if not os.path.isdir(collection_folder):
                return jsonify({"error": "Invalid collection folder path"}), 400

            size = int(request.form["size"])
            distance = int(request.form["distance"])
            color_enhance = float(request.form["color_enhance"])
            source_overlay = float(request.form["source_overlay"])
            color_effect = request.form["color_effect"]
            render_type = request.form["render_type"]

            result = create_photo_mosaic(
                main_photo,
                collection_folder,
                size,
                distance,
                color_enhance,
                source_overlay,
                color_effect,
                render_type,
            )

            # Save the mosaic image
            mosaic_path = os.path.join("static", "mosaic.jpg")
            result.save(mosaic_path, "JPEG", quality=95)

            # Create deep zoom images
            deep_zoom_dir = os.path.join("static", "deep_zoom")
            if os.path.exists(deep_zoom_dir):
                shutil.rmtree(deep_zoom_dir)
            os.makedirs(deep_zoom_dir)
            create_deep_zoom_tiles(result, deep_zoom_dir)

            return jsonify(
                {
                    "mosaic_url": "/static/mosaic.jpg",
                    "deep_zoom_url": "/static/deep_zoom/dzc_output.dzi",
                }
            )
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 400

    return render_template("index.html")


@app.route("/static/deep_zoom/<path:filename>")
def serve_deep_zoom(filename):
    return send_file(os.path.join("static", "deep_zoom", filename))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
