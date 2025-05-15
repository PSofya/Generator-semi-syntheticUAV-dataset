import cv2
import numpy as np

class AtmosphericEffects:
    @staticmethod
    def apply_fog(img, params):
        """Apply physically accurate fog effect with automatic depth estimation
        Args:
            img: Input image (BGR format)
            params: Dictionary containing:
                - intensity: Fog density (default 0.6)
                - color: Fog color in RGB (default (0.9, 0.9, 0.9))
                - focal_length: Camera focal length for depth estimation (default 1000)
                - probability: Effect application chance [0-1]
        Returns:
            Foggy image or original if skipped
        """
        if not params['enable'] or np.random.rand() > params['probability']:
            return None
        # Get parameters with defaults
        fog_intensity = params.get('intensity', 0.6)
        fog_color = params.get('color', (0.9, 0.9, 0.9))
        focal_length = params.get('focal_length', 1000)

        # Calculate depth map
        depth_map = AtmosphericEffects.estimate_ground_depth(img, focal_length)

        # Convert image to float
        img_float = img.astype(np.float32) / 255.0

        # Calculate transmission
        distance = 1.0 - depth_map
        transmission = np.exp(-fog_intensity * distance)

        # Apply fog effect
        fog_color = np.array(fog_color[::-1])  # Convert to BGR
        fogged = img_float * transmission[..., np.newaxis] + fog_color * (1 - transmission[..., np.newaxis])

        return (np.clip(fogged, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def estimate_ground_depth(image, focal_length=1000):
        """Improved depth estimation with sky detection
        Args:
            image: Input image (BGR format)
            focal_length: Virtual camera focal length in pixels
        Returns:
            Normalized depth map (0-1)
        """
        if image is None:
            raise ValueError("Input image is None")

        # Sky detection via HSV color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sky_mask = (hsv[:,:,1] < 50) & (hsv[:,:,2] > 200)

        # Initialize depth map
        height, width = image.shape[:2]
        depth = np.zeros((height, width))

        # Perspective depth calculation
        x = np.arange(width) - width/2
        y = np.arange(height) - height/2
        xx, yy = np.meshgrid(x, y)
        camera_depth = np.sqrt(focal_length**2 + xx**2 + yy**2)

        # Process ground areas
        ground_rows, ground_cols = np.where(~sky_mask)
        if ground_rows.size > 0:
            min_row, max_row = np.min(ground_rows), np.max(ground_rows)

            # Vertical depth component
            row_depth = 1 - (ground_rows - min_row) / (max_row - min_row)

            # Perspective component normalization
            cam_depth = camera_depth[ground_rows, ground_cols]
            cam_depth_norm = (cam_depth - cam_depth.min()) / (cam_depth.max() - cam_depth.min())

            # Combined depth
            depth[ground_rows, ground_cols] = 0.7*row_depth + 0.3*(1 - cam_depth_norm)

        # Process sky areas
        depth[sky_mask] = depth[~sky_mask].min() * 0.5

        # Final normalization
        return (depth - depth.min()) / (depth.max() - depth.min())
