# Development of Deep Learning Models for Cephalometric Landmark Localization

## Abstract

Cephalometric landmarks play a crucial role in medical and dental applications, facilitating diagnostics and treatment planning. Manual localization of these landmarks is time-consuming and prone to human error. This thesis focuses on the development of deep learning models to automate the localization of cephalometric landmarks.

### Models Implemented

Four deep learning models were implemented and evaluated for this task:

- Stacked Hourglass (SHG) with and without intermediate supervision
- YOLOv3
- U-Net

Among these models, YOLOv3 demonstrated superior adaptability, primarily due to its regression-based loss function. The other models introduced the challenge of generating 2D Gaussian heatmaps around keypoints, affecting landmark prediction accuracy.

### Evaluation Metrics

The evaluation methodology followed the ISBI 2015 Grand Challenge, which set the benchmark for cephalometric landmark localization. Key success parameters included the mean Euclidean distance for each landmark in millimeters and the percentage of predicted points within an allowable error.

### Results

Results for the four models:

- SHG: Mean Relative Error (MRE) - 15.65%
- SHG with Intermediate Supervision: MRE - 11.31%
- U-Net: MRE - 6.75%
- YOLOv3: MRE - 1.87%

Landmark accuracy within specified distances (in millimeters):

| Model                | <2mm | <3mm | <4mm |
|----------------------|------|------|------|
| Stacked Hourglass    | 4.36 | 10.40| 16.91|
| SHG with Intermediate Supervision | 8.81 | 19.05| 30.21|
| U-Net                | 33.18| 42.54| 48.50|
| YOLOv3               | 67.66| 87.71| 94.57|

### Dataset

The dataset used comprises 725 cephalometric images, each annotated with 22 cephalometric landmarks. It is essential to note that the dataset is private, respecting privacy and ethical considerations. For model training and evaluation, a 10% subset of these images was used.

### Codebase

This thesis emphasizes not only the development and evaluation of deep learning models but also the creation of a comprehensive codebase for cephalometric landmark localization. The codebase includes the implementation of the four models, along with essential data preprocessing and evaluation scripts. It is intended to serve as a valuable resource for researchers and professionals in the field, providing a foundation for future projects and advancements in cephalometric landmark localization. The codebase will be made publicly available on a platform like GitHub to encourage access and collaboration within the research community.

For more details, please refer to the complete thesis documentation.
