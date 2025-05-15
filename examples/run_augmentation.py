import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auglib.core.processor import AugmentationProcessor
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'src', 'data', 'uav_dataset')
    output_dir = os.path.join(base_dir, 'src', 'data', 'uav_dataset_augmented')

    processor = AugmentationProcessor(
        input_dir=input_dir, # It will write your path to \uav_dataset
        output_dir=output_dir # It will write your path to dataset augmented
    )
    # Process custom effects
    processor.process_images()
    # Process Augmentor pipeline
    processor.run_augmentor_pipeline()
    # Save final metadata
    processor.save_metadata()

if __name__ == "__main__":
    main()