

##  Repository Structure

The workflow is organized into two main chapters:

### Chapter 1: Data Acquisition & Multi-Dimensional Analysis
* `Chapter 1_API Web Scraper.ipynb`: Automated collection of street view and OSMnx data.
* `Chapter 1_API Interaction.ipynb`: Leveraging Gemini 2.5 Flash for semantic image description.
* `Chapter 1_Machine Learning_Gaze.ipynb`: Visual saliency analysis using DeepGaze III.
* `Chapter 1_Machine Learning_Yolo & SAMs.ipynb`: Object detection and instance segmentation for urban features.
* `Chapter 1_Visualising.ipynb`: Statistical EDA (Brightness, Saturation, Composition Style).

### Chapter 2: Latent Space Mapping & Cross-Modal Search
* `Chapter 2_Dataset Vectorisation_Text.ipynb`: Lemmatization and CLIP encoding for novel texts.
* `Chapter 2_Dataset Vectorisation_IMAGE.ipynb`: Visual feature extraction for paintings.
* `Chapter 2_Monitoring SOM Training (QE & TE).ipynb`: Training quality control via Quantization and Topographic Error.
* `Chapter 2_PCA-Based SOM Visualisation.ipynb`: RGB-mapped semantic visualization.
* `Chapter 2_Cross-Modal Search.ipynb`: **Final Integration.** 10-probe retrieval and seamless mosaic SOM generation.

---

##  Setup & Dependencies

### Environment
* **Python**: 3.11 or higher
* **Hardware**: CUDA-enabled GPU recommended for DeepGaze and CLIP.

### Installation
```bash
# Core Data & Visualization
pip install pandas numpy matplotlib seaborn tqdm Pillow scikit-learn

# Machine Learning Frameworks
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
pip install ultralytics  # For YOLO & SAMs
pip install git+[https://github.com/matthias-k/DeepGaze.git](https://github.com/matthias-k/DeepGaze.git)
pip install minisom      # For SOM training
