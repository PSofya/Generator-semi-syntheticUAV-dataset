# Generator-semi-syntheticUAV-dataset
The generator is designed to synthesize realistic distortions in sets of aerial photographs taken from UAVs. The system implements a modular approach to transformations, ensuring reproducibility and control of augmentation parameters.
#### Note: The effects of "fisheye" and wide-angle lens), geometric and photometric transformations and atmospheric conditions (fog overlay) are implemented.

## 0. Table of Contents

* [Introduction](#1-introduction)

* [About Generator](#2-about-generator)

* [Data Source](#3-data-source)

* [Contributions](#4-contributions)

* [Contacts](#5-contacts)

## 1. Introduction
In conditions of limited availability of GNSS and the dynamic nature of the environment, the creation of semi-synthetic UAV image datasets capable of
reproducing real-world capture conditions becomes a critical task. Currently,
there are various methods for generating semi-synthetic data, many of which are designed for tasks unrelated to aerial photography. 
The proposed generator expands the possibilities of creating semi-synthetic data by combining real satellite images with artificially
modeled conditions (changes in lighting, weather, optical distortions). 
The implementation was carried out using Python and specialized libraries, and the generated data were validated through SSIM metrics, as well as visual and statistical analyses.
The results confirmed a high preservation of key visual features and the adequacy of the modeling approach. 
## 2. About Generator
### Architecture Overview  
The semi-synthetic data generator is designed as a **modular system** that ensures sequential processing of input images and preserves metadata linking augmented results to their original frames.  

---

### Core Requirements  
- **Modularity**: Each transformation is implemented as an independent function.  
- **Error Handling**: The system issues warnings for unprocessable images without interrupting the generation pipeline.  

---

### Generator Modules  

#### 1. **Input/Output Module**  
- **Input**: Raw UAV images (JPEG format) stored in the `uav_dataset` directory.  
- **Output**:  
  - Augmented images saved to the `uav_dataset_augmented` directory.  
  - Metadata (CSV format) linking augmented images to their originals and documenting applied transformations.  

#### 2. **Optical Distortion Module**  
Simulates UAV-specific optical effects:  
- **Fisheye distortion**
- **Wide-angle distortion**  

#### 3. **Photometric & Geometric Augmentation Module**  
Applies transformations with configurable parameters:  
- **Photometric**:  
  - Brightness, contrast, color balance adjustments  
- **Geometric**:  
  - Rotations, flips, scaling  
- **Parameters**:  
  - Application probability for each transformation  
  - Customizable intensity ranges  

#### 4. **Fog Simulation Module**  
Physically accurate fog rendering based on:  
- Depth estimation  
- Light scattering models  
- Transmission coefficient calculations  

#### 5. **Metadata Generation**  
Stores per-image metadata in CSV format:  
- Original image
- Applied transformations with parameters  
- Geospatial data:  
  - Image center coordinates  
  - Frame width,frame height 
  - Camera tilt angles
### File structure:
```
Generator-semi-syntheticUAV-dataset/
├── requirements.txt
├── setup.py
├── src/
│   └── data/
│       ├── uav_dataset/                /* Raw Images
│           ├── metadata.csv            /* format as: filename,center_x,center_y,angle_deg,frame_width,frame_height
│           ├── frame_0000.jpg
│           ├── frame_0005.jpg
│           ├── frame_0010.jpg
|           ...
│       └── uav_dataset_augmented/      /* Transformed Images
│           ├── augmented_metadata.csv  /* format as:  filename,center_x,center_y,angle_deg,frame_width,frame_height,applied_effect,parameters
│           ├── frame_0000_fisheye.jpg    /* fisheye effect image
│           ├── frame_0000_wideangle.jpg  /* image with wide angle effect
│           ├── uav_dataset_original_frame_0000.jpg_11ef3b38-f4ae-467d-8c6a-3b55a4c3cb28.jpg   /* geometric transformation of image
│           ├── frame_0005_fisheye.jpg
|           ...
├── config/
│   ├── __init__.py
│   └── augmentations.py    /* augmentation parameters
├── auglib/
│   ├── __init__.py
│   ├── effects/
│   │   ├── __init__.py
│   │   ├── optical.py      /* fisheye and wide angle effects
│   │   └── atmospheric.py  /* fog
│   └── core/
│       ├── __init__.py
│       └── processor.py    /* applying geometric transformations, load and save metadata, applying augmentations
└── examples/
    └── run_augmentation.py /* main
```
---

### Implementation Details  
- **Language**: Python  
- **Core Libraries**:  
  - OpenCV (image processing)  
  - NumPy (numerical operations)  
  - Augmentor (geometric transformations)  
- **Metadata Format**: CSV for seamless integration with ML training/validation pipelines  

---

## 3. Data Source
This repository generates synthetic UAV images from **high-resolution TIFF maps** (Aklavik and UAV-VisLoc datasets).  

---

#### **TIFF Map Processing**  
- **Key Features**:  
  - Multi-layer support for complex visual analysis  
  - High-resolution imagery with embedded metadata:  
    - Geolocation coordinates  
    - Capture timestamps  
    - Sensor parameters  
    - Layer-specific properties  
  - Alpha channel preservation for advanced compositing  

---

#### **UAV Image Synthesis Process**  
1. **TIFF Cropping**:  
   - Extract region-of-interest using embedded geospatial metadata  
   - Preserve original coordinate system and projection  

2. **Overview Image Generation**:  
   - Create low-resolution previews for dataset exploration  

3. **UAV-Style Image Synthesis**:  
   - Slice large TIFF maps into UAV-sized fragments  
   - Export as JPEG/PNG with resolution mimicking real UAV sensors  

4. **Metadata Preservation**:  
   - Inherit original TIFF metadata  
   - Add synthetic camera parameters
---

#### **Implementation**  
The core processing workflow is implemented in the script located at: [processing_tiff.py](https://github.com/PSofya/Generator-semi-syntheticUAV-dataset/blob/main/src/scripts/processing_tiff.py).

## 4. Contributions
The importance of this study lies in addressing the limitations of current visual localization methods that often struggle in complex environments. The modular design of the generator ensures **reproducibility, scalability, and compatibility** with downstream computer vision workflows.
The results demonstrates that training models on the generated semi-synthetic data could lead to **improvements in localization accuracy**.

## 4. Contacts
Developer: saborisova_1@miem.hse.ru (Borisova Sofya)

a.romanov@hse.ru (Aleksandr Romanov, Professor)

https://www.hse.ru/en/staff/a.romanov#sci

https://www.researchgate.net/profile/Aleksandr-Romanov

