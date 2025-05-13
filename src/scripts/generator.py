import numpy as np
import cv2
import os
import pandas as pd
import Augmentor
from datetime import datetime

# ========== Configuration ==========
INPUT_DATASET = "/uav_dataset"
OUTPUT_DATASET = "/uav_dataset_augmented"
METADATA_PATH = os.path.join(INPUT_DATASET, "metadata.csv")
AUGMENTATION_PARAMS = {
    'contrast': {'probability': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
    'rotation': {'probability': 0.7, 'max_left_rotation': 15, 'max_right_rotation': 15},
    'brightness': {'probability': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
    'color': {'probability': 0.5, 'min_factor': 0.8, 'max_factor': 1.2},
    'hist_eq_prob': 0.3,
    'flip_lr_prob': 0.5,
    'flip_tb_prob': 0.2
}

# ========== Processing Functions ==========
def validate_image(img):
    """Validate minimum image size"""
    if img is None:
        raise ValueError("Image is None")
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        raise ValueError(f"Image too small: {h}x{w}")

def apply_effect(img, effect_name, **params):
    """Apply specific effect"""
    if effect_name == "fisheye":
        return fisheye_power(img, **params)
    elif effect_name == "wideangle":
        return apply_wideangle_effect(img, **params)
    elif effect_name == "fog":
        depth_map = estimate_ground_depth(img)
        return apply_physical_fog(img, depth_map, **params)
    else:
        raise ValueError(f"Unknown effect: {effect_name}")

# ========== Fish-eye Effect Function ==========
def fisheye_power(img, strength=2.0):
    """Apply fish-eye effect without saving file"""
    height, width = img.shape[:2]

    # Create coordinate grid
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)

    # Convert to polar coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Apply distortion
    r_normalized = r / np.max(r)
    r_distorted = r_normalized ** strength
    r_distorted *= np.max(r)

    # Convert back to Cartesian coordinates
    x_distorted = r_distorted * np.cos(theta)
    y_distorted = r_distorted * np.sin(theta)

    # Normalize coordinates
    x_distorted = ((x_distorted + 1) * width / 2).astype(np.float32)
    y_distorted = ((y_distorted + 1) * height / 2).astype(np.float32)

    # Remap image
    distorted_img = cv2.remap(img, x_distorted, y_distorted,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return distorted_img

def apply_wideangle_effect(img, k1=-0.2, k2=0.1, k3=0.0, k4=0.0):
    """Applies wide-angle lens effect (opposite of fisheye)"""
    h, w = img.shape[:2]

    # Camera matrix setup
    fx = fy = w * 0.8  # Smaller focal length for wider FOV
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]])

    # Distortion coefficients (k1 < 0 for pincushion effect)
    D = np.array([k1, k2, k3, k4])

    # Generate distortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )

    # Apply distortion
    wide_img = cv2.remap(img, map1, map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    return wide_img

def calculate_uav_depth(image_height, image_width, focal_length=1000):
    """Create depth map based on camera geometry"""
    # Create coordinate grid
    x = np.arange(image_width) - image_width/2
    y = np.arange(image_height) - image_height/2
    xx, yy = np.meshgrid(x, y)

    # Calculate distance from camera center
    depth = np.sqrt(focal_length**2 + xx**2 + yy**2)

    # Normalize and invert (1 = closest points)
    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
    return depth

def estimate_ground_depth(image, focal_length=1000):
    """Improved depth estimation with sky detection"""
    # Input validation
    if image is None:
        raise ValueError("Input image is None")

    # Sky detection via HSV color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sky_mask = (hsv[:,:,1] < 50) & (hsv[:,:,2] > 200)

    # Depth map initialization
    height, width = image.shape[:2]
    depth = np.zeros((height, width))

    # Perspective depth calculation
    x = np.arange(width) - width/2
    y = np.arange(height) - height/2
    xx, yy = np.meshgrid(x, y)
    camera_depth = np.sqrt(focal_length**2 + xx**2 + yy**2)

    # Ground objects processing
    ground_rows, ground_cols = np.where(~sky_mask)
    if len(ground_rows) > 0:
        min_row, max_row = np.min(ground_rows), np.max(ground_rows)

        # Vertical depth component
        row_depth = 1 - (ground_rows - min_row) / (max_row - min_row)

        # Perspective component normalization
        cam_depth = camera_depth[ground_rows, ground_cols]
        cam_depth = (cam_depth - cam_depth.min()) / (cam_depth.max() - cam_depth.min())

        # Combined depth
        depth[ground_rows, ground_cols] = 0.7*row_depth + 0.3*(1-cam_depth)

    # Sky processing
    depth[sky_mask] = depth[~sky_mask].min() * 0.5

    # Final normalization
    return (depth - depth.min()) / (depth.max() - depth.min())

def apply_physical_fog(image, depth_map, fog_intensity=0.6, fog_color=(0.9, 0.9, 0.9)):
    """Apply physically accurate fog effect"""
    # Image validation and conversion
    if image is None:
        raise ValueError("Input image is None")

    img_float = image.astype(np.float32) / 255.0

    # Convert depth to distance
    distance = 1.0 - depth_map

    # Transmission coefficient calculation
    transmission = np.exp(-fog_intensity * distance)

    # Fog effect creation
    fog_color = np.array(fog_color[::-1])  # Convert to BGR
    fogged = img_float * transmission[..., np.newaxis] + fog_color * (1 - transmission[..., np.newaxis])
    return (np.clip(fogged, 0, 1) * 255).astype(np.uint8)

# ========== Modified Processing Code ==========
def generate_effect_versions(img_path, output_dir, metadata_df):
    """Generate augmented versions with different effects"""
    effects = {
        "fisheye": {"strength": 1.2},
        "wideangle": {"k1": -0.5, "k2": 0.05},
        "fog": {"fog_intensity": 0.8}
    }
    
    try:
        img = cv2.imread(img_path)
        validate_image(img)
        fname = os.path.basename(img_path)
        new_entries = []

        for effect_name, params in effects.items():
            # Apply effect
            if effect_name == "fisheye":
                processed_img = fisheye_power(img, **params)
            elif effect_name == "wideangle":
                processed_img = apply_wideangle_effect(img, **params)
            elif effect_name == "fog":
                depth_map = estimate_ground_depth(img)
                processed_img = apply_physical_fog(img, depth_map, **params)

            # Saving and metadata
            new_fname = f"{os.path.splitext(fname)[0]}_{effect_name}.jpg"
            output_path = os.path.join(output_dir, new_fname)
            cv2.imwrite(output_path, processed_img)
            
            orig_meta = metadata_df[metadata_df['filename'] == fname].iloc[0].copy()
            orig_meta['filename'] = new_fname
            orig_meta['applied_effect'] = effect_name
            new_entries.append(orig_meta)

        return new_entries

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return []

def main():
    # Initialization
    os.makedirs(OUTPUT_DATASET, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    all_new_metadata = []

    # Effects processing
    for fname in os.listdir(INPUT_DATASET):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(INPUT_DATASET, fname)
            new_entries = generate_effect_versions(img_path, OUTPUT_DATASET, metadata)
            all_new_metadata.extend(new_entries)

    # Augmentor processing
    p = Augmentor.Pipeline(INPUT_DATASET, OUTPUT_DATASET)
    p = configure_augmentor(p)
    
    def augmentor_callback(image, _):
        """Metadata callback for Augmentor"""
        new_fname = f"augmentor_{datetime.now().strftime('%f')}.jpg"
        orig_meta = metadata[metadata['filename'] == image.file_name].iloc[0].copy()
        orig_meta['filename'] = new_fname
        orig_meta['applied_effect'] = "augmentor_pipeline"
        all_new_metadata.append(orig_meta)
        return image

    p.sample(30)

    # Save full metadata
    full_metadata = pd.concat([metadata, pd.DataFrame(all_new_metadata)], ignore_index=True)
    full_metadata.to_csv(
        os.path.join(OUTPUT_DATASET, "augmented_metadata.csv"),
        index=False
    )

def configure_augmentor(pipeline):
    """Configure Augmentor pipeline"""
    p = pipeline
    p.random_contrast(**AUGMENTATION_PARAMS['contrast'])
    p.rotate(**AUGMENTATION_PARAMS['rotation'])
    p.flip_left_right(AUGMENTATION_PARAMS['flip_lr_prob'])
    p.flip_top_bottom(AUGMENTATION_PARAMS['flip_tb_prob'])
    p.random_brightness(**AUGMENTATION_PARAMS['brightness'])
    p.random_color(**AUGMENTATION_PARAMS['color'])
    p.histogram_equalisation(AUGMENTATION_PARAMS['hist_eq_prob'])
    return p

if __name__ == "__main__":
    main()
    print("All augmentations completed with metadata tracking!")
