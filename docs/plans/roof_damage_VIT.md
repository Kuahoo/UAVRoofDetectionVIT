# Technical Implementation Plan: Automated UAV Roof Damage Detection via Vision Transformers and U-Net

**DISCLAIMER: This document outlines an academic machine-learning prototype for informational and research purposes only. The resulting system is strictly not to be used for certified structural inspections, professional safety assessments, or any safety-critical evaluations.**

## Executive Summary

This document outlines the end-to-end technical implementation plan for a machine-learning-based roof defect inspection prototype using Unmanned Aerial Vehicle (UAV) imagery. The core engineering objective is to automate the detection and localization of roof defects, addressing the inefficiencies and hazards of manual inspections. The system employs a dual-architecture deep learning approach: a Vision Transformer (ViT) for multi-label classification of the **five defect types** (identifying *what* is present) and a U-Net for six-class pixel localization including background (identifying *where*). The imbalanced RoofD dataset requires weighted `BCEWithLogitsLoss` for the ViT and a weighted CE + Dice hybrid for the U-Net so rare classes such as patches and breaks are not ignored. A logistic regression baseline anchors Macro F1 on the same five defect labels. Success is Macro F1 (ViT/baseline) and mIoU (U-Net). The Streamlit demo fuses both pipelines with five-class confidence scores and U-Net pixel overlays for a human-in-the-loop inspector.

*   **Dual-Architecture Approach:** The system utilizes a Vision Transformer (ViT) for image-level multi-label classification to detect defect presence, running parallel to a U-Net architecture for pixel-level defect localization.
*   **Primary Focus Designation:** The primary technical responsibility detailed in this plan is the engineering, tuning, and evaluation of the Vision Transformer pipeline, utilizing the PyTorch Image Models (`timm`) library.
*   **Partner Alignment:** A secondary, distinct pipeline relies on the `segmentation_models_pytorch` library to instantiate the U-Net model; this domain is clearly demarcated for partner development.
*   **Constraint Management:** The implementation introduces targeted mathematical and procedural interventions to handle the severe class imbalance inherent in the target dataset, alongside experiments simulating degraded Unmanned Aerial Vehicle (UAV) image resolutions. 

This document serves as the comprehensive, end-to-end technical implementation plan for developing a machine-learning-based roof defect inspection prototype. The primary engineering problem addresses the inefficiency and hazard of manual roof inspections by utilizing automated inference on aerial RGB imagery. To accomplish this, the project leverages the public RoofD dataset. Because analyzing high-resolution drone imagery involves distinct tasks—knowing *what* is damaged versus *where* it is damaged—this plan bifurcates the machine learning architecture. The focus of this specific document heavily details the classification pipeline utilizing a Vision Transformer, while rigorously establishing the architectural boundaries, shared data pipelines, and evaluation frameworks required to integrate smoothly with the U-Net semantic segmentation model developed by the designated team partner.

## 1. Locked Project Scope and Architectural Division

To ensure a realistic and achievable semester project, the scope is strictly constrained to building a non-autonomous inspection assistant prototype. The system is designed to highlight potential roof damage for human review rather than issuing certified structural assessments. 

The technical architecture is divided into three distinct machine learning algorithms, distributed among team members to satisfy course requirements while constructing a cohesive final application.

First, to answer the question of what defect types appear in the roof image, the project will implement a **Vision Transformer (ViT)**. This model will perform multi-label image classification, meaning it will output independent probabilities for the presence of multiple overlapping defect classes (e.g., an image can contain both a "hollow" and a "break"). This component represents the primary focus of this detailed technical plan.

Second, to answer the question of exactly where the defects are located, a partner will implement a **U-Net** semantic segmentation model. This model will classify every individual pixel in the image into one of the designated defect categories or the background.

Finally, to fulfill the requirement for a course-taught baseline, both members will collaborate on a **Logistic Regression** model utilizing handcrafted image features. This serves strictly as a quantitative anchor to measure the performance gains achieved by the deep learning architectures. 

## 2. Dataset Engineering and Processing Pipeline

The foundation of both the ViT and U-Net pipelines is a robust, shared data engineering framework. Ensuring that data is loaded, transformed, and batched identically is critical for maintaining experimental integrity when comparing model performances.

### 2.1 The RoofD Dataset Characteristics

The project exclusively utilizes the **RoofD dataset**, a public repository specifically engineered for flat-roof defect segmentation from aerial imagery, compiled by Yudin et al. [cite: 1, 2]. The public archive is approximately **6.2 GB** and contains aerial RGB images with corresponding pixel-level segmentation masks [cite: 2]. (`scripts/download_roofd.py` records the exact expected byte size.)

The dataset provides a total of 6,400 image pairs, pre-divided into a training set of 5,184 images and a test set of 1,216 images [cite: 1, 2]. The native resolution for all images is 1024 × 576 pixels [cite: 1, 2]. Pixel masks use six categories: **background** plus five defect types—**hollows, swellings, folds, patches, and breaks** [cite: 1, 3].

A critical engineering constraint of RoofD is its severe class imbalance. While background, hollows, and swellings dominate, patches and breaks are exceedingly rare, with patches appearing in only 107 training images and breaks in a mere 54 training images [cite: 2]. 

This severe imbalance creates a realistic engineering challenge. Physically, a "break" often blends seamlessly into the chaotic texture of surrounding roof gravel or is obscured by natural shadows, making the defect visually indistinguishable from normal structural wear. Similarly, small "patches" can mimic the color profile of accumulated dirt or puddles [cite: 3, 4]. As a result, these specific defects are visually difficult for models to detect in drone imagery, necessitating specialized loss functions and sampling techniques to prevent the models from defaulting to majority-class predictions.



### 2.2 Ground-Truth Mask Processing and Label Generation

The provided dataset utilizes RGB color masks to denote different defect classes. To make this data usable for PyTorch models, the data loading pipeline must deterministically map these RGB values to integer class indices. 

The exact color mapping provided by the RoofD dataset must be codified into the data loader. The background is mapped to `(0, 0, 0)`, hollows to `(255, 255, 0)`, swellings to `(153, 255, 204)`, folds to `(6, 89, 182)`, patches to `(0, 255, 255)`, and breaks to `(255, 193, 255)` [cite: 2]. 

During the dataset initialization phase, the custom PyTorch `Dataset` class will execute the following routine for every image-mask pair:
1. Load the RGB aerial image and the RGB segmentation mask using OpenCV or Pillow.
2. Convert the 3-channel RGB mask into a single-channel 2D integer map with values 0–5 (background through breaks). This map is the U-Net ground-truth target.
3. Derive the Vision Transformer / logistic **defect presence** target by scanning which of the five defect IDs (1–5) appear in that map. Emit a length-**5** multi-label binary vector in the order `[hollows, swellings, folds, patches, breaks]`. Background is **excluded**: it is almost always present in every image, so a sixth “background” classifier node would be near-constant, inflate Macro F1, and dilute gradients for rare defects. Example: an image with swellings and a break yields `[0, 1, 0, 0, 1]`.

Both models still share the same decoded mask and the same spatial augmentations; they differ only in how that mask is summarized (pixel map vs. defect-presence vector).

### 2.3 Preprocessing and Augmentation Configuration

To standardize inputs and manage computational costs, images will be aggressively downscaled during the standard training loop. Configurations will be maintained centrally in a `configs/default.yaml` file to ensure reproducibility.

The Vision Transformer will expect an input resolution of 224 × 224 pixels, which aligns with standard pre-trained architectures [cite: 5]. The U-Net will expect inputs of 256 × 256 pixels to maintain clean downsampling and upsampling divisibility within the convolutional blocks [cite: 6].

After resizing, RGB images for both deep models are scaled to `[0, 1]` and normalized with **ImageNet** statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). This is required for transfer learning from ImageNet-pretrained `timm` / SMP weights; omitting it would make the frozen and fine-tuned features miscalibrated.

Data augmentation will be strictly limited to spatial transformations that do not alter the semantic meaning of the roof defects. These include random horizontal flips (`p=0.5`), random vertical flips (`p=0.5`), and random rotations (restricted to 90-degree increments with `p=0.5`) to avoid complex mask interpolation artifacts. The same geometric transform is applied to the image and its mask so classifier labels and segmentation maps stay aligned. Because the Vision Transformer lacks the inherent translation invariance of Convolutional Neural Networks (CNNs), these augmentations are particularly critical to prevent the ViT from overfitting to the spatial locations of defects within the training set.

## 3. Primary Focus: Vision Transformer (ViT) Implementation

The core technical responsibility of this plan is the design and execution of the Vision Transformer pipeline. Unlike traditional CNNs that process images through sliding convolutional kernels, a ViT treats an image as a sequence of flattened patches. By applying **Self-Attention Mechanisms**—a core mathematical operation that dynamically weighs the importance of every part of the input data relative to every other part, regardless of their spatial distance—the ViT can model global dependencies across the entire roof surface simultaneously. 

Think of this like a reader interpreting a complex word based on the entire context of a sentence rather than just looking at adjacent letters. In the context of roof inspection, this means the ViT can correlate a small, dispersed water stain in one corner of the roof with a subtle fold on the opposite side, potentially identifying contextual clues that a localized convolutional filter might miss.

### 3.1 Model Instantiation via TIMM

To avoid the fragility and overhead of writing a Vision Transformer from scratch, the implementation will rely on **TIMM (PyTorch Image Models)**. TIMM is a comprehensive library providing over 1,600 pre-trained image models and a standardized API that dramatically simplifies transfer learning [cite: 7, 8].

The specific architecture selected for this task is `vit_base_patch16_224` [cite: 5]. This model breaks the 224 × 224 input image into a grid of 16 × 16 pixel patches (196 patch tokens plus the class token).

The model will be instantiated with `timm.create_model`, using `pretrained=True` for ImageNet weights and **`num_classes=5`** for the five defect labels [cite: 5]. That call replaces the original 1000-class head with an uninitialized linear layer of shape `(embed_dim, 5)`. Background remains a U-Net-only class and must not appear in this head.

### 3.2 Adapting to Multi-Label Classification

A standard pre-trained ViT expects a multi-class problem (exactly one label) and typically applies Softmax. Roof images can contain multiple simultaneous defects (e.g., folds and swellings), so this task is **multi-label** [cite: 9, 10].

Training uses `BCEWithLogitsLoss`, which applies Sigmoid internally for numerical stability. At inference and for metrics, the five raw logits are passed through an independent **Sigmoid** so each value in `[0, 1]` is the probability that the corresponding defect is present. Softmax is never used.

### 3.3 Loss Function and Class Imbalance Mitigation

Standard categorical cross-entropy is invalid for multi-label outputs. Optimization uses **`BCEWithLogitsLoss`** [cite: 10] over the five defect logits.

Severe imbalance (patches and breaks are rare) will otherwise drive the model to predict “absent” for minority classes. Countermeasure: compute a length-5 `pos_weight` vector from **training-split image counts only** (never from val/test). For each defect class $c$,

\[
\text{pos\_weight}_c = \frac{N_{\text{neg},c}}{N_{\text{pos},c}}
\]

where $N_{\text{pos},c}$ is the number of training images containing class $c$. Pass this tensor into `BCEWithLogitsLoss(pos_weight=...)`. Rare defects then incur larger loss when missed, so attention must learn their appearance.

### 3.4 Fine-Tuning Strategy and Optimization

Training a ViT on ~5,184 images risks catastrophic forgetting if the full network is unlocked immediately. Use staged fine-tuning with AdamW (`weight_decay=0.01`), gradient clipping at `1.0`, and AMP on CUDA:

**Phase 1 (warm-up, 5 epochs):** freeze all backbone parameters; train only the new 5-node head with AdamW at learning rate `1e-3`. Calibrates the head without corrupting pretrained attention.

**Phase 2 (deep tuning, 15 epochs):** reload the best warm-up validation checkpoint if one was saved; unfreeze the **final two** transformer blocks plus the head; AdamW at `1e-5` with **CosineAnnealingLR** over the 15 epochs. Keep the checkpoint with the best validation **Macro F1 over the five defect classes**, then reload it and report test metrics.

## 4. Partner Focus: U-Net Semantic Segmentation

While the Vision Transformer identifies the presence of defects, it lacks the architectural ability to draw precise bounding boxes or masks. This requirement is fulfilled by the partner's domain: the U-Net semantic segmentation model.

The U-Net architecture is optimized for tasks requiring both fine detail and contextual awareness, making it an ideal choice for identifying visually subtle roof defects like folds or swellings [cite: 11]. It achieves this through an encoder path that downsamples the image to capture abstract features, and a decoder path that upsamples the features to recover spatial dimensions, connected by "skip connections" that preserve fine-grained spatial information [cite: 12].

### 4.1 Implementation via Segmentation Models PyTorch (SMP)

To match the rapid deployment capability of the ViT pipeline, the partner will utilize the `segmentation_models_pytorch` (SMP) library [cite: 11]. SMP provides a high-level API that constructs complex segmentation networks using pre-trained classification backbones as encoders [cite: 13, 14, 15, 16].

The U-Net will be instantiated using `smp.Unet`. The partner will configure the initialization with `encoder_name="resnet34"` and `encoder_weights="imagenet"` [cite: 11, 14, 17]. The input channels will be set to 3 (for RGB images), and the output classes will be set to 6 [cite: 14, 16, 18].

Relying on ResNet34 as the encoder provides a lightweight but highly capable convolutional backbone that converges in approximately 10 fewer epochs than a standard U-Net trained from scratch [cite: 13, 19].

### 4.2 Segmentation Loss Strategy

Unlike the multi-label ViT, the U-Net performs multi-class pixel classification—meaning every single pixel must belong to exactly one of the six categories (inclusive of the background). Therefore, the U-Net will output a tensor of shape `(Batch, 6, Height, Width)`.

The standard loss function for this architecture is Pixel-wise Cross-Entropy Loss. However, because the background class dominates 90% of the pixel area in the RoofD dataset, Cross-Entropy alone will result in a model that simply predicts "background" everywhere.

To resolve this, the partner's pipeline will utilize a hybrid loss function combining Weighted Cross-Entropy and **Dice Loss**. The Dice Loss mathematically optimizes for the Intersection over Union, directly penalizing the network for failing to overlap with the minority defect pixels [cite: 3, 14]. The combination of these two loss functions ensures the network learns clear boundaries (via Cross-Entropy) while aggressively seeking out rare defect instances (via Dice Loss).

### 4.3 U-Net Optimization and Fine-Tuning Strategy

To ensure structural parity with the ViT pipeline, the U-Net model will follow a defined optimization framework. The model will be trained using the standard Adam optimizer with an initial learning rate of 1e-4. The network will be trained for a total of 20 epochs. To manage convergence and prevent overfitting, a `ReduceLROnPlateau` scheduler will be utilized, programmed to reduce the learning rate by a factor of 0.1 if the validation Mean Intersection over Union (mIoU) fails to improve for 3 consecutive epochs. This structured approach allows the ResNet34 encoder to steadily adapt its pre-trained ImageNet weights to the specific geometric structures of roof defects without aggressively destroying learned feature maps.

## 5. Course-Taught Baseline: Logistic Regression

To satisfy academic requirements and establish a quantifiable baseline, a traditional machine learning pipeline will be implemented jointly. The selected algorithm is Logistic Regression utilizing handcrafted feature extraction. 

The baseline pipeline strips away the complexities of deep learning to test whether simple visual heuristics can detect roof damage. The pipeline operates as follows:
1. The 1024 × 576 RGB image is severely downscaled to 64 × 64 pixels to make feature extraction computationally viable on CPUs.
2. The image is converted to grayscale.
3. A **Histogram of Oriented Gradients (HOG)** algorithm is applied to extract texture and edge density features. Flat roofs generally have uniform gradients; defects like swellings and folds disrupt this uniformity.
4. The HOG feature matrix is flattened into a 1D vector and concatenated with basic color statistics (mean and variance of the original RGB channels).
5. This 1D feature vector is fed into a One-vs-Rest Logistic Regression classifier implemented via `scikit-learn` using the `lbfgs` solver, a maximum of 1,000 iterations, and a regularization parameter C-value of 1.0 to prevent overfitting.

The baseline predicts the **same five defect labels** as the ViT and is scored with the same Macro F1. Expect roughly usable Macro F1 on color-salient defects (e.g., patches) and near-failure on rare texture defects (e.g., breaks), which motivates the deep models.

## 6. Evaluation Framework and Metrics

Because the tasks of classification and segmentation are mathematically distinct, they cannot be evaluated using a single shared accuracy metric. Furthermore, standard global accuracy is highly misleading on the RoofD dataset due to the severe class imbalance; a model predicting "background" for every image would achieve over 80% accuracy while failing completely at defect detection.

### 6.1 Classification Metrics (ViT and Baseline)

ViT and logistic predictions (Sigmoid probabilities over the **five defect classes**) are thresholded at 0.5 for presence/absence.

The primary metric is **Macro F1** over those five classes only: per-class F1 (precision/recall harmonic mean) averaged with equal weight so rare “breaks” count the same as common “hollows.” Background is never included in classification Macro F1.

Also log exact-match (subset) accuracy and per-class precision/recall. For multi-label diagnostics, report each class’s **2×2** counts (TN/FP/FN/TP) from `multilabel_confusion_matrix`—not a single 5×5 mutually exclusive confusion matrix, which would be invalid when multiple defects co-occur.

### 6.2 Segmentation Metrics (U-Net)

The partner's U-Net model will be evaluated based on spatial overlap. 

The primary metric will be **Mean Intersection over Union (mIoU)**. For each class, the IoU is calculated by dividing the area of overlap between the predicted mask and the ground truth by the total area encompassed by both. The mIoU is the average of these class-specific scores. (For reference against published standards on this dataset, HRNet+OCR reached 0.44 mIoU and U-NetMCT reached 0.33-0.37 mIoU [cite: 1, 3]).

Additionally, the **Dice Score** will be reported, which places slightly more emphasis on true positives, making it a standard evaluation metric for imbalanced medical and aerial segmentation tasks.

### 6.3 Model Specifications Summary

To explicitly clarify the distinct roles and mathematical foundations of the three implemented algorithms, the following table summarizes their core specifications:

| Model Architecture | Task Type | Input Resolution | Loss Function | Output Format | Primary Evaluation Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Transformer (ViT)** | Multi-Label Image Classification | 224 × 224 + ImageNet norm | Weighted `BCEWithLogitsLoss` | 1D Tensor (5 defect probabilities) | Macro F1 (5 defects) |
| **U-Net (ResNet34 Backbone)** | Semantic Pixel Segmentation | 256 × 256 + ImageNet norm | Dice Loss + Weighted Cross-Entropy | 2D Tensor (6-class pixel map) | Mean Intersection over Union (mIoU) |
| **Logistic Regression (Baseline)** | Multi-Label Image Classification | 64 × 64 grayscale HOG + RGB stats | Log Loss (one-vs-rest) | 1D Tensor (5 defect probabilities) | Macro F1 (5 defects) |

### 6.4 Hardware Requirements and Operational Logistics (TIDE)

**All full model training runs on SDSU TIDE** (cluster details in `docs/context/TIDE.md`).

TIDE provides 17 GPU nodes with **4× NVIDIA L40 (48 GB)** each (68 L40s total) plus one scarce **GPU-Advanced** node with 4× A100 (80 GB). CPU-only nodes exist for non-GPU work. The **default interactive allocation is 8 CPU cores, 32 GB system RAM, and one L40**. Prefer the L40 for this project; do not request the A100 unless L40 capacity is unavailable—the ViT/U-Net footprints do not need 80 GB.

**How to run:**
- **Interactive / short iteration:** JupyterHub with the **pytorch-notebook** image (Ubuntu 22.04, Python 3.11, CUDA `torch` / `torchvision` preinstalled). Install the remaining project deps with `pip install -r requirements.txt`.
- **Long training / overnight:** submit a **batch container** job on the cluster so the run is not tied to a browser session.
- Store RoofD (~6.2 GB archive) and `outputs/` on TIDE Ceph-backed storage so downloads and checkpoints persist across sessions.

**Batch and throughput settings** (locked for reproducibility in `configs/default.yaml`):
- ViT (`224×224`): batch size **16**, `num_workers: 8` (matches default core count), AMP on CUDA.
- U-Net (`256×256`): batch size **8**, same worker/AMP policy.
- These sizes were originally sized for ≥8 GB VRAM and are therefore comfortable on the L40; raising batches is optional for speed experiments but must be reported if changed.

Under the default TIDE L40 allocation, expect on the order of a few minutes per epoch for each deep model (exact time depends on contention and I/O), which supports rapid iteration during fine-tuning.

## 7. Engineering Experiments: Simulating UAV Constraints

To elevate the project beyond a basic modeling exercise, the final phase introduces an engineering experiment reflecting real-world UAV operational constraints. 

UAVs operating at higher altitudes cover more roof area per flight minute but capture lower-resolution imagery of the underlying defects. The experimental question is: *How much image detail is required before small roof defects become unreliable to detect?*

Once the ViT and U-Net models have achieved baseline convergence on their respective 224 × 224 and 256 × 256 inputs, a secondary test protocol will be initiated. The original 1024 × 576 test set images will be artificially degraded (downsampled using bilinear interpolation—a resampling method that uses the distance-weighted average of the four nearest pixel values to estimate a new pixel color—and then upsampled back to standard input sizes) to simulate lower resolution captures. 

The pipeline will evaluate model performance on simulated resolutions of 512 × 288 and 256 × 144. The expected outcome is a nonlinear degradation in recall for the rare, physically small "break" and "patch" classes, while the larger "swelling" classes remain resilient. This experiment provides a tangible, operational recommendation regarding maximum allowable drone flight altitudes for automated inspections.

## 8. Application Deployment via Streamlit

The culmination of the project is an interactive demonstration application built using the **Streamlit** framework. The goal of this application is not to build a production-ready enterprise tool, but to prove the viability of the ML pipeline in a human-in-the-loop inspection scenario.

The application script (`src/app.py`) will orchestrate the dual-model inference pipeline. The user workflow is designed for simplicity:
1. The user uploads an aerial RGB roof image through the Streamlit web interface.
2. The backend dynamically resizes the image to 224 × 224 for the ViT and 256 × 256 for the U-Net, applying the necessary normalizations.
3. The Vision Transformer runs first. The UI shows Sigmoid probabilities for the **five defect classes** and lists detections at a 50% threshold (background is never shown as a ViT class).
4. Simultaneously, the U-Net produces a 2D segmentation mask over all six pixel classes.
5. OpenCV blends the U-Net mask over the original image with a semi-transparent RoofD color overlay.
6. The interface shows the original image beside the overlay so the inspector sees both *what* and *where*.

Because the ViT and U-Net are independent, disagreements are expected. Conflict logic is deterministic and **never invents pixels**:
- **Unlocalized Confidence Warning** — ViT ≥ 50% for a defect, U-Net has zero pixels of that class; warn in text, keep the U-Net overlay unchanged.
- **Sub-threshold Spatial Detection** — U-Net has defect pixels, ViT confidence < 50%; still show the mask and warn in text.

A hardcoded disclaimer will be rendered below the results explicitly stating: *"For research demonstration only; professional inspection is required."*

## 9. Timeline and Milestone Tracking

To ensure completion within the semester constraints, the project strictly adheres to a phased milestone structure. 

*   **Milestone 1: Data Pipeline Integrity.** Shared data framework complete: RoofD downloaded (~6.2 GB archive), verified, split with `seed=42`. Loaders yield length-5 defect multi-label targets for ViT/logistic and length-6 integer masks for U-Net from the same decoded RGB masks.
*   **Milestone 2: Baseline Execution.** Logistic baseline trains on the five defect labels; Macro F1 is recorded under `outputs/`.
*   **Milestone 3: Vision Transformer Convergence (Primary Owner).** `timm` `vit_base_patch16_224` with `num_classes=5`, ImageNet normalization, weighted `BCEWithLogitsLoss`, and staged fine-tuning. Validation Macro F1 (5 defects) exceeds the logistic baseline.
*   **Milestone 4: U-Net Convergence (Partner Owner).** The `segmentation_models_pytorch` U-Net trains without tensor shape mismatches. The model generates coherent prediction overlays, and the mIoU metrics are successfully logged.
*   **Milestone 5: Engineering Constraints Validated.** The secondary test script applying the resolution degradation experiment is run against both models, generating the required figures and data tables for the final report.
*   **Milestone 6: Streamlit Integration and Final Deliverables.** The `app.py` script successfully imports both trained `.pt` checkpoint files and executes inference on an uploaded image without crashing. The final report is compiled, explicitly documenting the individual contributions of the ViT and U-Net implementations, alongside the presentation slide deck and the required 1–3 minute video demonstration.

**Sources:**
1. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFylCGG-lX5SY1MkJTkhOUXZFX43vJuTIfJvkUtQCNoANekHXCGHfra_zON_De3YnOqWMc0ieLsbyKMsfjcTsmSBJaDDT_L7IGWoRZliim9Ttw3swAte0M_wXLqE4oVzd8Q5ugszwLxrlFWwf2DtZiImmXyd8mCWDBItG70BCdlH-dhtSZBOfTMP0TtDSCgupzODOLeN7t2f_vVKNjILPdwEYt7tDtGcqBKbKiM0v6twrozSoreDg==)
2. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFEswG82O9lZMb85jeobcHiL8a5KE3k2GiaQVO3sNUMnH6kK5dING3zHgMNnzf-6WlBQ8b3V1N8t9Z9yUdHG9gYSwNwFwDbaQWYFBc-GGNuDPLOTF0=)
3. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG9TGIUVcsmOoe0FpKeylWuPOxuOzMS_9Ua3InxlOoKiAr8OrGkcfNPCxafj0xyQ88A3K_vSrWQksgu1rUAOaHLsi9vRbtZJgQujwNEAxH977d5CVhz6UnMIed_rXgoPUfRrJTltyIzI9Ph0GEecoov7JsD5pkw6O-zeP7GgmLBCrokDPBuAowc8iKD_Sc5BaMGIOqRHqjOY6FUtC6yxUwcBd5hONSgFw==)
4. [colab.ws](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGaiyLjnwFzIdn38bJmZm8iywMaSRvLkaqld9-IcpbzbFLImDM7NxtcNuzY0Os6B9tbKKyYYg_aLvamZqYBy-zq9Qg81YYRlhHKx62jA8Yk-55rr6v50mBGdLvPmi8guFg6zazN9v-BC54Otkqq)
5. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHcK2Q7BPCJkOC-vZ0cKV7co-eXm-ta9zHd0hyojjY8Tnbsyxe-YIF8IAk09UmwgV9-znfxzacP4Hw4WS3kbXra8WKDhkiSCPnIjbgwDF36j6CkvjSRFaoch5GtK6yQ9o3-IgLnufNK-qMjLA==)
6. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGfbhhX5EqFBol5jvuoCIJQyNyQu9da3zaxToVZoDwkNuYzX7gzZmHlcSgpWMCEMUvuQDqX-rMO_DSjhBL3kQYncgw7OdA5FonWCTboqyDSRWyhHiML620LOYwQ3qo-Qqk=)
7. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEhNhW0om1zOI3pA6nrosznY6oy4VjsGPXyY6Tx5_NakIF0FPWGTRNt0Du6xuTqULIUZWtrHmY36N7CExlwluXiVffNYHlWgbU7ADrKzzHAZw==)
8. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFsyTr9tzdxbkKftzh5Uu7VJ-RmNCvkKQqTqE4M9L6yAESJigWaskUNwkB9EKbCEBhf15l5IzaUdMyMZsFXnYg-BQp0iviGx1g25y9MUFtPV7D2v6HJhUBD9FTiYauUovQTEQR7gQicjwdYIsdhugIWu0KvT5N2qGs4PNXK82SUOhDR3gvMWfePXOcYG8eWqpD2r9Kbpm4PGgjpCqD8sr1Rv2OF1lVb053IAp9_YH9O2-KmD2QzfN8fS13QHg==)
9. [google.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFczgTPPVM3YqInsJ3rR9V-J-Q2iYxnjELQIBdZvAQFUop-nCkp5yAwLP-BQMJ5Q1gETfNgLcWa-Si1QIxA6iZMEp0TkbrzTtLYBw3HwNVHnRxxsJ5Q91Q6aQ3GBaCrFA2x2Uwuz435sIXB77vZ2LtoXPnex1GXSXEEikBuxkRHK6onTzcIFpTmZXsCO9c28Q3y-VEN787Bd8ZSEFEpE4u8PDMKRGwsatx0lFGVUfsmlcexO_UZY53cvq3hfw3ZIwOnkmO14HWsejbhje_-wXj-RmV7oFYkk9Fu9wCK2ARlIeGGaH2RcgsMcrF1kA==)
10. [kaggle.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGeJ4WxqVUuEfoIV0NKqSJeM4mqdFPnw38vlfYi1EbkNRHXMZYzqJZIyQStKC7YAEFAGSQ9l68rrzVCUuL32zzGipvEOLTt8Ge6kNHyc6Luzdd0tTlgiDYsF8QRTpkoVwy6t_tCtIM4MDp3zNMfc-56YCmciILYL5uGyl5Jh1tiCLKtae7P7vNikOam)
11. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG4FqDl9HL-M_gRaFbiEdt8YWRe_Rz40l4D9gohFZAOrAyJ1SXyYm4S1gbDeMAEiCPnFeUcLW5eiztdbFwTiDvKSA8sWtyOEZgD6HoNmCvsENlcmijQ1DQn0ihRuUvdtlQXfqaLq-PQ5yJrb5IeSQYl145Rhuf-AI67OdlifQzU-7uYSdT_q-uzgpd6hw8=)
12. [readthedocs.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFhQbdy5x_JLw_y8qInH26nEvPBVbjwzIY2LR_76UmrakqO_hefa0h2Fv3WbSzWtOPepVKhIGs-1d3HtbbwuJdPmeYleq-t-R0fJkPvCoUNK0QfL54ZhgCamDyGhBIzVUlx9pOWog==)
13. [pypi.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEjY8lk4lXb2C-SoERxO9XdZz9bsfQ285ucyJvpASseco0uyULXVpVDwBN6cnCbIqXh3k7NRFM9Ogv3okXOUvS0W7_dB0DIpnTivO1fk6DUNnGESD-nljI8etDDtKUNYJdrE7CK_9C5Pfo5VLsHH1yd)
14. [pypi.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG_DwGnxjEG_cR08SK7OwFvDH7eymjfcLZgEwbTN7quPbj4zc9mzSfSmM5t7J3ThrmuQ8XmWy8JvMzgafzROwF1_DwgcmQTocT6woYgnr7vzaDS9tC_IhP0SBkPGMKR_LYoqNMQa7UnLDyq)
15. [readthedocs.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFaTUhTfkpHaJk-6eyhRh9mDfZGail22v_MOz9qirqK21pDXEhOS_xM84M3Bmx0-CcXxggtvP8YCordOtNfcqfi0AO4AvwoV_UXGdFE4TC7mhOXKMYkwJ-p-19Gf7Ktbp2lz1yqGA3-W__IMu7P90Kk9g==)
16. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFlyNnDSK7aTEeDJZFAQo0Hjwrme7vD-4pNFk2BupHRcN4S7pf-egu1dG1vPjUv9Xi_bfFaFXMa9eghGmWgmRjcJJNhyTIpDYFdGdb5kk3Ud2IXVviYCdL2n1YZZvxJkFjhBVPj8E0WD1AELJM=)
17. [readthedocs.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEAlHFa7VfSdEtHiwHWOKN45s-xTilRctc6PNKmCIOcoe5NUs850sFMoRl1vsBv3mzZ0PcTXq2913lgNKBR-WoZza6BI8IaWIV-BrF63hxN7U4dDVZKpFQqgPTJXRMjX5WfRea2lA==)
18. [readthedocs.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGHM5Kt9eN0TTdZz83BN7dN2UNl1RTV5BngQ7_cP_2-ufTCXNITLqpMgFW-8NZeuRThPJbRssdSpH1_qZG14x809KAsmJS6Y7WXaw0LVgFq3gi9oxjZ8kOqxeaPe6CJ-yzH92PbXmEPXsEqqNtGyp_bjuC9dKxfJ69VUpUXzqf9ykI=)
19. [context7.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHEll3mk8jXXQcbZvu-M4nsYUyXR_L5TSsS8jANyu32p2WEVkDjU6Ifa_OzQs5HYyiQDRKIhs8PRJAh5bSL6RkjX-VznXxnLeEuHuIQ0nr3NyGF3I0JOucBfLWIcTT67x9RvqiFx2rbcYuJtf2E_TIB)

