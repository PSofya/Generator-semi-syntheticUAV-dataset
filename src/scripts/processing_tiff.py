import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from lightglue import LightGlue, SuperPoint, ALIKED, DISK
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
from math import radians, sin, cos, sqrt, asin

def create_overview(input_tiff, output_png, meta_path):
    """Generate satellite.png from cropped TIFF with metadata"""
    with rasterio.open(input_tiff) as src:
        # Maintain original resolution ratio
        scale_factor = min(3840/src.width, 2160/src.height)
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.cubic
        )
        
        # Save metadata
        meta = pd.read_csv(meta_path).iloc[0]
        pd.DataFrame([{
            'original_crop_left': meta['original_left'],
            'original_crop_top': meta['original_top'],
            'pixel_size_x': meta['resolution'],
            'pixel_size_y': meta['resolution'],
            'overview_width': new_width,
            'overview_height': new_height
        }]).to_csv(output_png.replace('.png', '_meta.csv'), index=False)

        # Convert and save image
        data = np.moveaxis(data, 0, -1)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(output_png, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
def crop_tiff_with_metadata(input_path, output_path, meta_path,
                           x_offset, y_offset, width, height):
    """Crop TIFF and save geospatial metadata"""
    with rasterio.open(input_path) as src:
        window = Window(x_offset, y_offset, width, height)

        # Get original map bounds
        orig_bounds = src.bounds
        orig_res = src.res[0]

        # Crop and save
        data = src.read(window=window)
        transform = src.window_transform(window)

        meta = src.meta.copy()
        meta.update({
            "height": height,
            "width": width,
            "transform": transform
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)

    # Save metadata CSV
    pd.DataFrame([{
        'original_map': Path(input_path).name,
        'x_offset_pix': x_offset,
        'y_offset_pix': y_offset,
        'crop_width': width,
        'crop_height': height,
        'original_left': orig_bounds.left,
        'original_right': orig_bounds.right,
        'original_top': orig_bounds.top,
        'original_bottom': orig_bounds.bottom,
        'resolution': orig_res
    }]).to_csv(meta_path, index=False)
def pixel_to_wgs84(x_pixel, y_pixel):
    """Use CROPPED metadata for conversion"""
    meta = pd.read_csv("satellite_cropped_meta.csv").iloc[0]
    lon = meta['original_crop_left'] + x_pixel * meta['pixel_size_x']
    lat = meta['original_crop_top'] - y_pixel * meta['pixel_size_y']
    return lon, lat

# Configuration
INPUT_TIFF = output_tiff
METADATA_CSV = output_meta
OUTPUT_DIR = path_02+"/uav_dataset"
IMAGE_SIZE = (3976, 2652)  # (width, height)
START_DATE = "2018-10-23T08:32:25"
BASE_HEIGHT = 466.52  # meters


def generate_uav_images():
    # Load cropping metadata
    meta = pd.read_csv(METADATA_CSV).iloc[0].to_dict()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Initialize metadata list
    metadata = []
    current_date = datetime.fromisoformat(START_DATE)

    with rasterio.open(INPUT_TIFF) as src:
        # Calculate geographic parameters
        res = meta['resolution']  # degrees per pixel
        x_offset = meta['x_offset_pix']
        y_offset = meta['y_offset_pix']

        # Generate sliding window coordinates
        for i, y in enumerate(range(0, src.height, IMAGE_SIZE[1])):
            for j, x in enumerate(range(0, src.width, IMAGE_SIZE[0])):
                # Create window
                window = Window(x, y, IMAGE_SIZE[0], IMAGE_SIZE[1])

                # Skip partial windows
                if x + IMAGE_SIZE[0] > src.width or y + IMAGE_SIZE[1] > src.height:
                    continue

                # Read and convert image data
                data = src.read(window=window)
                data = np.moveaxis(data, 0, -1)  # CHW -> HWC

                # Convert to 8-bit
                if data.dtype != np.uint8:
                    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                # Calculate center coordinates
                center_x = x + IMAGE_SIZE[0] // 2
                center_y = y + IMAGE_SIZE[1] // 2

                # Convert to original map coordinates
                orig_x = x_offset + center_x
                orig_y = y_offset + center_y
                # Calculate WGS84 coordinates
                lon = meta['original_left'] + (x_offset + center_x) * meta['resolution']
                lat = meta['original_top'] - (y_offset + center_y) * meta['resolution']

                # Generate filename
                filename = f"syn_{len(metadata)+1:04d}.jpg"
                filepath = Path(OUTPUT_DIR) / filename

                # Save image (convert RGB to BGR for OpenCV)
                cv2.imwrite(str(filepath), cv2.cvtColor(data, cv2.COLOR_RGB2BGR))

                # Generate synthetic metadata
                metadata.append({
                    'num': len(metadata)+1,
                    'filename': filename,
                    'date': current_date.isoformat(),
                    'lat': round(lat, 8),
                    'lon': round(lon, 8),
                    'height': BASE_HEIGHT + np.random.uniform(-10, 10),
                    'Omega': np.random.uniform(-1, 1),
                    'Kappa': np.random.uniform(-1, 1),
                    'Phi1': np.random.uniform(-40, -50),
                    'Phi2': np.random.uniform(-50, -40)
                })

                # Increment timestamp by 2 seconds
                current_date += timedelta(seconds=2)

    # Save metadata
    pd.DataFrame(metadata).to_csv(path_02+"/synthetic_metadata.csv", index=False)


# First: Create cropped TIFF and metadata
path_02="/content/drive/MyDrive/4 курс/вкр/Localization/02"
output_tiff = path_02+"/cropped_center2.tif"
output_meta = path_02+"/cropped_center2_metadata.csv"

with rasterio.open(path_map) as src:
    full_width, full_height = src.width, src.height
    crop_width, crop_height = 8775, 6685
    
    crop_tiff_with_metadata(
        input_path=path_map,
        output_path=output_tiff,
        meta_path=output_meta,
        x_offset=int(full_width/2 - crop_width/2),
        y_offset=int(full_height/2 - crop_height/2),
        width=crop_width,
        height=crop_height
    )

# Second: Create overview from CROPPED TIFF
create_overview(
    input_tiff=output_tiff,
    output_png=path_02+"/satellite_cropped.png",
    meta_path=output_meta
)

# Third: Generate UAV images from the SAME cropped TIFF
generate_uav_images()


# Configuration
SAT_META_PATH = path_02+"/satellite_cropped_meta.csv"
UAV_META_PATH = path_02+"/synthetic_metadata.csv"
SUCCESS_THRESHOLD = 10  # meters


# Load geospatial metadata
sat_meta = pd.read_csv(SAT_META_PATH).iloc[0]

# Load ground truth from UAV metadata
uav_meta = pd.read_csv(UAV_META_PATH)
true_row = uav_meta[uav_meta['filename'] == 'syn_0001.jpg'].iloc[0]
true_lon = true_row['lon']
true_lat = true_row['lat']


# --- New Metric Calculation Section ---
def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between GPS coordinates in meters"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371000 * 2 * asin(sqrt(a))

# Estimate position
if len(m_kpts0) >= 4:
    # Find transformation matrix
    M, inliers = cv2.estimateAffinePartial2D(
        m_kpts0.cpu().numpy(),
        m_kpts1.cpu().numpy(),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )
    
    if M is not None:
        # Transform UAV image center to satellite coordinates
        h, w = ov.shape[:2]
        center = np.array([w/2, h/2, 1])
        pred_pix = M @ center
        
        # Convert to WGS84
        pred_lon = sat_meta['original_crop_left'] + pred_pix[0] * sat_meta['pixel_size_x']
        pred_lat = sat_meta['original_crop_top'] - pred_pix[1] * sat_meta['pixel_size_y']
        
        # Calculate metrics
        error = haversine(true_lon, true_lat, pred_lon, pred_lat)
        success = error <= SUCCESS_THRESHOLD
    else:
        error = float('inf')
        success = False
else:
    error = float('inf')
    success = False

# Visualization
axes = viz2d.plot_images([ov, img])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
viz2d.add_text(0, f'Error: {error:.2f}m | Success: {success}', fs=16)

print(f"\nLocalization Metrics:")
print(f"- RMSE: {error:.2f} meters")
print(f"- Success ({SUCCESS_THRESHOLD}m): {'Yes' if success else 'No'}")
print(f"- Valid Matches: {len(m_kpts0)}/{len(matches)}")