import cv2
import numpy as np

class OpticalEffects:
    @staticmethod
    def fisheye(img, params):
        """Apply fish-eye effect with configurable parameters
        Args:
            img: Input image (numpy array)
            params: Dictionary containing:
                - strength: Distortion strength (default 2.0)
                - probability: Application probability [0-1]
        Returns:
            Distorted image or original if skipped
        """
        if not params['enable'] or np.random.rand() > params['probability']:
            return None
        strength = params.get('strength', 2.0)
        height, width = img.shape[:2]

        # Create normalized coordinate grid
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)

        # Convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Apply radial distortion
        # Normalize r to be between 0 and 1
        r_normalized = r / np.max(r)
        r_distorted = r_normalized ** strength
        r_distorted *= np.max(r)

        # Convert back to Cartesian coordinates
        x_distorted = r_distorted * np.cos(theta)
        y_distorted = r_distorted * np.sin(theta)

        # Map to image coordinates
        x_distorted = ((x_distorted + 1) * width / 2).astype(np.float32)
        y_distorted = ((y_distorted + 1) * height / 2).astype(np.float32)

        # Remap the image using the distorted coordinates
        fisheye_img = cv2.remap(img, x_distorted, y_distorted,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        return fisheye_img

    @staticmethod
    def wideangle(img, params):
        """Apply wide-angle (pincushion) distortion
        Args:
            img: Input image (numpy array)
            params: Dictionary containing:
                - k1: Main distortion coefficient (negative for wide-angle)
                - k2: Secondary distortion coefficient
                - probability: Application probability [0-1]
        Returns:
            Distorted image or original if skipped
        """
        if not params['enable'] or np.random.rand() > params['probability']:
            return None
        h, w = img.shape[:2]
        k1 = params.get('k1', -0.2)
        k2 = params.get('k2', 0.1)
        k3 = params.get('k3', 0)
        k4 = params.get('k4', 0)

        # Camera matrix
        fx = fy = w * 0.8 # Slightly smaller focal length for wider FOV
        cx, cy = w/2, h/2
        K = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]], dtype=np.float32)

        # Distortion coefficients (k1 negative for pincushion effect)
        D = np.array([k1, k2, k3, k4], dtype=np.float32)

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
