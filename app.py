import os
import io
import math
import random
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template, send_file, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_average_color(image):
    """Calculate the average color of an image."""
    img_array = np.asarray(image)
    return tuple(int(x) for x in img_array.mean(axis=(0, 1)))


def apply_color_effect(image, effect):
    """Apply a color effect to an image."""
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


def process_cell_image(image_path):
    """Process and resize an image for use in the mosaic."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            img = img.crop((left, top, right, bottom))
            img = img.resize((64, 64), Image.LANCZOS)
        return img
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None


def detect_aspect_ratio(image):
    """Detect the closest standard aspect ratio for an image."""
    width, height = image.size
    ratios = [(1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16)]
    closest_ratio = min(ratios, key=lambda r: abs(width / height - r[0] / r[1]))
    return closest_ratio


def crop_to_aspect_ratio(image, aspect_ratio):
    """Crop an image to match the specified aspect ratio."""
    width, height = image.size
    target_ratio = aspect_ratio[0] / aspect_ratio[1]
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    elif current_ratio < target_ratio:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))

    return image


def add_watermark(
    image,
    banner_text="Sample made using www.picmyna.com. The paid file wouldn't have these watermarks",
    repeated_text="Sample Image",
    bottom_banner_text="www.picmyna.com",
):
    """Add watermark to the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Top Banner Watermark
    banner_text_width, banner_text_height = draw.textsize(banner_text, font=font)
    padding = 10
    banner_width = image.width
    banner_height = banner_text_height + 2 * padding
    banner_color = (0, 0, 0, 128)
    banner = Image.new("RGBA", (banner_width, banner_height), banner_color)
    draw_banner = ImageDraw.Draw(banner)
    banner_text_position = ((banner_width - banner_text_width) // 2, padding)
    draw_banner.text(
        banner_text_position, banner_text, font=font, fill=(255, 255, 255, 255)
    )
    image.paste(banner, (0, 0), banner)

    # Bottom Banner Watermark
    bottom_banner_height = banner_text_height + padding * 2
    bottom_banner_width = image.width
    bottom_banner_color = (0, 0, 0, 128)
    bottom_banner = Image.new(
        "RGBA", (bottom_banner_width, bottom_banner_height), bottom_banner_color
    )
    draw_bottom = ImageDraw.Draw(bottom_banner)
    bottom_text_width, bottom_text_height = draw_bottom.textsize(
        bottom_banner_text, font=font
    )
    text_x = (bottom_banner_width - bottom_text_width) // 2
    text_y = (bottom_banner_height - bottom_text_height) // 2
    draw_bottom.text(
        (text_x, text_y), bottom_banner_text, font=font, fill=(255, 255, 255, 255)
    )
    image.paste(bottom_banner, (0, image.height - bottom_banner_height), bottom_banner)

    # Repeated Watermark
    draw = ImageDraw.Draw(image)
    repeated_text_width, repeated_text_height = draw.textsize(repeated_text, font=font)
    for y in range(0, image.height, repeated_text_height * 4):
        for x in range(0, image.width, repeated_text_width * 4):
            draw.text((x, y), repeated_text, font=font, fill=(255, 255, 255, 64))

    return image.convert("RGB")


def create_photo_mosaic(
    main_photo,
    collection_folder,
    size,
    distance,
    color_enhance,
    source_overlay,
    color_effect,
    is_preview=True,
):
    """Create a photo mosaic from a main photo and a collection of images."""
    logger.info(
        f"Starting mosaic creation with parameters: size={size}, distance={distance}, color_enhance={color_enhance}, source_overlay={source_overlay}, color_effect={color_effect}"
    )

    start_time = time.time()

    logger.info("Opening and processing main image...")
    main_image = Image.open(main_photo).convert("RGB")
    aspect_ratio = detect_aspect_ratio(main_image)
    main_image = crop_to_aspect_ratio(main_image, aspect_ratio)
    logger.info(f"Main image processed. Aspect ratio: {aspect_ratio}")

    grid_sizes = {
        (1, 1): (60, 60),
        (4, 3): (60, 45),
        (3, 4): (45, 60),
        (3, 2): (60, 40),
        (2, 3): (40, 60),
        (16, 9): (64, 36),
        (9, 16): (36, 64),
    }

    if size == "small":
        grid_sizes = {k: (v[0] * 4 // 3, v[1] * 4 // 3) for k, v in grid_sizes.items()}
    elif size == "big":
        grid_sizes = {k: (v[0] * 2 // 3, v[1] * 2 // 3) for k, v in grid_sizes.items()}

    grid_width, grid_height = grid_sizes[aspect_ratio]
    cell_width, cell_height = (
        main_image.width // grid_width,
        main_image.height // grid_height,
    )
    logger.info(
        f"Grid size: {grid_width}x{grid_height}, Cell size: {cell_width}x{cell_height}"
    )

    logger.info("Processing collection images...")
    collection_images = []
    for filename in os.listdir(collection_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            img = process_cell_image(os.path.join(collection_folder, filename))
            if img:
                collection_images.append(img)
    logger.info(f"Processed {len(collection_images)} collection images")

    if not collection_images:
        raise ValueError("No valid images found in the collection folder")

    mosaic = Image.new("RGB", (main_image.width, main_image.height))
    main_array = np.asarray(main_image)

    total_cells = grid_width * grid_height
    image_count = len(collection_images)
    logger.info(f"Total cells: {total_cells}, Available images: {image_count}")

    # Adjust distance to work on a scale of 0-10
    if distance == 0:
        use_distance = False
        image_cycle = image_count
    else:
        use_distance = True
        image_cycle = min(image_count, max(1, int(image_count * (11 - distance) / 10)))

    logger.info(f"Using distance: {use_distance}, Image cycle: {image_cycle}")

    def process_cell(index):
        try:
            x = (index % grid_width) * cell_width
            y = (index // grid_width) * cell_height
            cell = main_array[y : y + cell_height, x : x + cell_width]
            avg_color = tuple(int(x) for x in cell.mean(axis=(0, 1)))

            if use_distance:
                image_index = index % image_cycle
            else:
                image_index = index % image_count

            best_match = collection_images[image_index]

            if color_enhance > 0:
                enhanced = Image.blend(
                    best_match,
                    Image.new("RGB", best_match.size, avg_color),
                    color_enhance / 100,
                )
            else:
                enhanced = best_match

            return (x, y, enhanced.resize((cell_width, cell_height), Image.LANCZOS))
        except Exception as e:
            logger.error(f"Error processing cell {index}: {str(e)}")
            return (x, y, Image.new("RGB", (cell_width, cell_height), avg_color))

    logger.info("Starting cell processing...")
    with ThreadPoolExecutor() as executor:
        future_to_cell = {
            executor.submit(process_cell, i): i for i in range(total_cells)
        }

        for future in as_completed(future_to_cell):
            try:
                x, y, cell_image = future.result()
                mosaic.paste(cell_image, (x, y))
            except Exception as e:
                logger.error(f"Error pasting cell image: {str(e)}")

    logger.info("Cell processing completed")

    if source_overlay > 0:
        logger.info(f"Applying source overlay: {source_overlay}%")
        mosaic = Image.blend(mosaic, main_image, source_overlay / 100)

    logger.info(f"Applying color effect: {color_effect}")
    mosaic = apply_color_effect(mosaic, color_effect)

    if is_preview:
        logger.info("Creating preview image")
        mosaic.thumbnail((1000, 1000), Image.LANCZOS)
        mosaic = add_watermark(mosaic)
    else:
        logger.info("Creating full-size image")
        max_size = 25000
        if mosaic.width > max_size or mosaic.height > max_size:
            mosaic.thumbnail((max_size, max_size), Image.LANCZOS)

    end_time = time.time()
    logger.info(f"Mosaic creation completed in {end_time - start_time:.2f} seconds")

    return mosaic


def create_deep_zoom_tiles(image, output_dir, tile_size=256, overlap=1):
    """Create deep zoom tiles for the mosaic image."""
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
    """Handle the main route for the application."""
    if request.method == "POST":
        try:
            main_photo = request.files["main_photo"]
            collection_folder = request.form["collection_folder"]
            size = request.form["size"]
            distance = int(request.form["distance"])
            color_enhance = float(request.form["color_enhance"])
            source_overlay = float(request.form["source_overlay"])
            color_effect = request.form["color_effect"]
            is_preview = request.form.get("watermark", "true").lower() == "true"
            print(request.form)
            if not os.path.isdir(collection_folder):
                return jsonify({"error": "Invalid collection folder path"}), 400

            # Generate a unique identifier for this request
            request_id = str(uuid.uuid4())

            logger.info(f"Creating mosaic for request {request_id}")
            logger.info(
                f"Parameters: size={size}, distance={distance}, color_enhance={color_enhance}, source_overlay={source_overlay}, color_effect={color_effect}, is_preview={is_preview}"
            )

            result = create_photo_mosaic(
                main_photo,
                collection_folder,
                size,
                distance,
                color_enhance,
                source_overlay,
                color_effect,
                is_preview,
            )

            # Save the mosaic image with a unique filename and also with the details of the request in the filename
            mosaic_filename = f"mosaic_{request_id}_______{size}__{distance}__{color_enhance}__{source_overlay}__{color_effect}__{'preview' if is_preview else 'final'}.jpg"
            mosaic_path = os.path.join("static", mosaic_filename)
            result.save(mosaic_path, "JPEG", quality=95)

            # Save the original image with a unique filename
            original_filename = f"original_{request_id}_______{size}__{distance}__{color_enhance}__{source_overlay}__{color_effect}__{'preview' if is_preview else 'final'}.jpg"
            original_path = os.path.join("static", original_filename)
            main_photo.save(original_path)

            # Create deep zoom images with a unique folder name
            deep_zoom_dir = os.path.join(
                "static",
                f"deep_zoom_{request_id}_______{size}__{distance}__{color_enhance}__{source_overlay}__{color_effect}__{'preview' if is_preview else 'final'}",
            )
            if os.path.exists(deep_zoom_dir):
                shutil.rmtree(deep_zoom_dir)
            os.makedirs(deep_zoom_dir)
            create_deep_zoom_tiles(result, deep_zoom_dir)

            return jsonify(
                {
                    "mosaic_url": f"/static/{mosaic_filename}",
                    "deep_zoom_url": f"{deep_zoom_dir}/dzc_output.dzi",
                    "original_url": f"/static/{original_filename}",
                }
            )
        except Exception as e:
            logger.error(f"Error creating mosaic: {str(e)}")
            return jsonify({"error": str(e)}), 400

    return render_template("index.html")


@app.route("/static/deep_zoom_<path:request_id>/<path:filename>")
def serve_deep_zoom(request_id, filename):
    """Serve deep zoom tiles."""
    return send_file(os.path.join("static", f"deep_zoom_{request_id}", filename))


def cleanup_old_files(directory, max_age_hours=1):
    """Remove files older than the specified age."""
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_hours * 3600:
                os.remove(file_path)
        elif os.path.isdir(file_path) and filename.startswith("deep_zoom_"):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_hours * 3600:
                shutil.rmtree(file_path)


@app.before_request
def cleanup_files():
    """Clean up old files before processing each request."""
    cleanup_old_files("static")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
