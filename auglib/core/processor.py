import os
import cv2
import pandas as pd
import Augmentor
import numpy as np
from datetime import datetime
from ..effects.optical import OpticalEffects
from ..effects.atmospheric import AtmosphericEffects
from config.augmentations import AUGMENTATION_PARAMS


class AugmentationProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.metadata = []
        self.original_metadata = pd.read_csv(
            os.path.join(input_dir, "metadata.csv")
        ) if os.path.exists(os.path.join(input_dir, "metadata.csv")) else pd.DataFrame()
        os.makedirs(output_dir, exist_ok=True)

    def _augmentor_callback(self, image):
        """Metadata handler for Augmentor output"""
        new_fname = f"augmentor_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"

        # Create new metadata entry
        new_meta = {
            'filename': new_fname,
            'applied_effect': 'augmentor_pipeline',
            'augmentor_params': str(AUGMENTATION_PARAMS['augmentor'])
        }

        # Try to find original metadata if possible
        if not self.original_metadata.empty:
            # Get random original metadata as fallback
            new_meta.update(self.original_metadata.sample(1).iloc[0].to_dict())

        new_meta['filename'] = new_fname  # Ensure filename is updated
        self.metadata.append(new_meta)
        return image

    def save_metadata(self):
        """Save only augmented images metadata to CSV"""
        if self.metadata:
            new_metadata_df = pd.DataFrame(self.metadata)
            new_metadata_df.to_csv(
                os.path.join(self.output_dir, "augmented_metadata.csv"),
                index=False
            )

    def _apply_effect(self, original_img, original_name, effect_name, effect_func):
        params = AUGMENTATION_PARAMS.get(effect_name, {})
        if not params.get('enable', False):
            return

        if np.random.rand() <= params.get('probability', 0):
            processed_img = effect_func(original_img.copy(), params)
            if processed_img is not None:
                base_name = os.path.splitext(original_name)[0]
                new_name = f"{base_name}_{effect_name}.jpg"
                cv2.imwrite(os.path.join(self.output_dir, new_name), processed_img)
                self._update_metadata(original_name, new_name, effect_name)

    def _add_augmentor_metadata(self, filename):
        """Add metadata entry for Augmentor-generated file"""
        new_meta = {
            'filename': filename,
            'applied_effect': 'augmentor_pipeline',
            'parameters': str(AUGMENTATION_PARAMS['augmentor'])
        }

        # Preserve original metadata fields if available
        if not self.original_metadata.empty:
            orig_meta = self.original_metadata.sample(1).iloc[0].to_dict()
            new_meta.update({k: v for k, v in orig_meta.items() if k != 'filename'})

        self.metadata.append(new_meta)
    @staticmethod

    def _load_metadata(self):
        metadata_path = os.path.join(self.input_dir, "metadata.csv")
        self.original_metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
    def _save_image(self, img, original_name, effect_name):
        """Save processed image and update metadata"""
        base_name = os.path.splitext(original_name)[0]
        new_name = f"{base_name}_{effect_name}.jpg"
        output_path = os.path.join(self.output_dir, new_name)

        cv2.imwrite(output_path, img)

        # Update metadata
        orig_meta = self.original_metadata[self.original_metadata['filename'] == original_name]
        if not orig_meta.empty:
            new_meta = orig_meta.iloc[0].copy()
            new_meta['filename'] = new_name
            new_meta['applied_effects'] = effect_name
            self.metadata.append(new_meta)
    def process_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            original_name = os.path.basename(img_path)

            # Apply effects independently
            self._apply_effect(img, original_name, 'fisheye', OpticalEffects.fisheye)
            self._apply_effect(img, original_name, 'wideangle', OpticalEffects.wideangle)
            self._apply_effect(img, original_name, 'fog', AtmosphericEffects.apply_fog)

            return True
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return False

    def _configure_augmentor(self, pipeline):
        """Configure Augmentor pipeline parameters"""
        p = pipeline
        params = AUGMENTATION_PARAMS['augmentor']

        if 'contrast' in params:
            p.random_contrast(**params['contrast'])
        if 'rotation' in params:
            p.rotate(**params['rotation'])
        if 'flip_lr_prob' in params:
            p.flip_left_right(params['flip_lr_prob'])
        if 'flip_tb_prob' in params:
            p.flip_top_bottom(params['flip_tb_prob'])
        if 'brightness' in params:
            p.random_brightness(**params['brightness'])
        if 'color' in params:
            p.random_color(**params['color'])
        if 'hist_eq_prob' in params:
            p.histogram_equalisation(params['hist_eq_prob'])

        return p


    def _update_metadata(self, original_name, new_name, effect_name):
        """Update metadata for custom effects"""
        if not self.original_metadata.empty:
            orig_meta = self.original_metadata[self.original_metadata['filename'] == original_name]
            if not orig_meta.empty:
                new_meta = orig_meta.iloc[0].copy()
                new_meta['filename'] = new_name
                new_meta['applied_effect'] = effect_name
                self.metadata.append(new_meta.to_dict())

    def run_augmentor_pipeline(self):
        """Execute Augmentor pipeline with metadata tracking"""
        if AUGMENTATION_PARAMS['augmentor']['enable']:
            # Create pipeline and configure operations
            p = Augmentor.Pipeline(self.input_dir, self.output_dir)
            p = self._configure_augmentor(p)

            # Track generated files and metadata
            generated_files = set(os.listdir(self.output_dir))

            # Execute pipeline
            p.sample(AUGMENTATION_PARAMS['augmentor'].get('samples', 30))

            # Find newly created files and add metadata
            new_files = set(os.listdir(self.output_dir)) - generated_files
            for fname in new_files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self._add_augmentor_metadata(fname)

    def process_images(self):
        """Process all images in input directory"""
        for fname in os.listdir(self.input_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.input_dir, fname)
                self.process_image(img_path)
