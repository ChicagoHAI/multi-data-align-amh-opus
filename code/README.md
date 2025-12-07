# Code Repositories for Multimodal Mental Health Research

This directory contains cloned repositories with implementations relevant to multimodal data alignment techniques for adolescent mental health AI modeling.

---

## Repository 1: Automatic Depression Detector

### Overview
- **Repository**: https://github.com/derong97/automatic-depression-detector
- **Location**: code/automatic-depression-detector/
- **Purpose**: Multi-model ensemble for automatic depression detection using DAIC-WOZ dataset
- **Modalities**: Text, Audio, Gaze features
- **Language**: Python

### Description
This project aims to detect if a participant is depressed using extracted text, gaze, and audio features from the DAIC-WOZ dataset. It implements an ensemble approach combining multiple models for improved depression detection accuracy.

### Key Features
- Feature extraction from DAIC-WOZ interviews
- Multi-modal fusion (text + audio + gaze)
- Ensemble learning approach
- Depression binary classification

### Key Files
- Feature extraction scripts
- Model training pipelines
- Evaluation metrics

### How to Use
1. Obtain DAIC-WOZ dataset (see datasets/README.md)
2. Extract features using provided scripts
3. Train models on multimodal features
4. Evaluate ensemble performance

### Dependencies
- Python 3.x
- TensorFlow/PyTorch (check requirements.txt)
- Audio processing libraries
- NLP libraries for text processing

### Notes
- Requires DAIC-WOZ dataset access
- Baseline implementation for depression detection
- Good starting point for multimodal mental health modeling

---

## Repository 2: Multimodal Depression from Video

### Overview
- **Repository**: https://github.com/cosmaadrian/multimodal-depression-from-video
- **Location**: code/multimodal-depression-from-video/
- **Purpose**: Non-verbal depression detection from videos using temporal models
- **Paper**: "Reading Between the Frames: Multi-Modal Non-Verbal Depression Detection in Videos"
- **Modalities**: Video (visual), Audio
- **Language**: Python

### Description
Official source code for detecting depression from non-verbal cues in videos. Proposes a simple and flexible multi-modal temporal model capable of discerning non-verbal depression cues from diverse modalities in noisy, real-world videos.

### Key Features
- Temporal modeling for video sequences
- Non-verbal cue extraction
- Multi-modal fusion of visual and audio features
- Robust to real-world noisy data

### Architecture
- Temporal models for sequence processing
- Multi-modal fusion strategies
- Depression severity prediction

### Key Files
- `models/`: Neural network architectures
- `data/`: Data loading and preprocessing
- `train.py`: Model training script
- `eval.py`: Evaluation scripts

### How to Use
1. Prepare video dataset (DAIC-WOZ or similar)
2. Extract features or use end-to-end pipeline
3. Train temporal model
4. Evaluate on test set

### Dependencies
- PyTorch
- OpenCV/ffmpeg for video processing
- Audio processing libraries

### Notes
- Focuses on non-verbal cues (vs. verbal content)
- Temporal modeling is key contribution
- Applicable to real-world video data

---

## Repository 3: Multi-Modal Depression Detection (ICASSP 2019)

### Overview
- **Repository**: https://github.com/genandlam/multi-modal-depression-detection
- **Location**: code/multi-modal-depression-detection/
- **Purpose**: Context-aware deep learning for multi-modal depression detection
- **Paper**: "Context Aware Deep Learning for Multi Modal Depression Detection" (ICASSP 2019, Oral)
- **Modalities**: Audio, Video, Text
- **Language**: Python

### Description
Official codebase for ICASSP 2019 paper on context-aware multimodal depression detection. Uses DAIC-WOZ dataset with deep learning models that incorporate contextual information from interviews.

### Key Features
- Context-aware modeling
- Multi-modal data fusion
- Deep learning architectures
- DAIC-WOZ dataset preprocessing

### Architecture
- Separate encoders for each modality
- Context-aware fusion mechanism
- Depression classification/regression

### Key Files
- Data preprocessing scripts for DAIC-WOZ
- Model implementations
- Training and evaluation pipelines
- Pre-trained model checkpoints (if available)

### How to Use
1. Request access to DAIC-WOZ dataset
2. Preprocess data using provided scripts
3. Train models with context-aware fusion
4. Evaluate performance metrics

### Dependencies
- Python 3.x
- Deep learning framework (TensorFlow/Keras)
- Audio/video processing libraries
- Natural language processing tools

### Notes
- Published at top-tier conference (ICASSP 2019)
- Context-aware approach is novel contribution
- Requires DAIC-WOZ dataset access
- Good reference for academic research

---

## Additional Useful Repositories (Not Cloned)

### 1. CLMLF - Contrastive Learning and Multi-Layer Fusion
- **Repository**: https://github.com/Link-Li/CLMLF
- **Paper**: NAACL 2022 Findings
- **Purpose**: Multimodal sentiment detection using contrastive learning
- **Why relevant**: Demonstrates contrastive learning for multimodal fusion

### 2. HAUCL - Hypergraph Autoencoder and Contrastive Learning
- **Repository**: https://github.com/zhziming/HAUCL
- **Purpose**: Emotion recognition in conversation
- **Why relevant**: Advanced fusion techniques applicable to mental health

### 3. PRISM - Passive Sensing for Mental Health
- **Repository**: https://github.com/wuami/prism
- **Purpose**: Passive, real-time information for sensing mental health
- **Why relevant**: Smartphone-based passive sensing framework

### 4. MultimodalGraph - Deep Graph Learning
- **Repository**: https://github.com/YongJiao10/MultimodalGraph
- **Purpose**: Treatment prediction in major depression using graph networks
- **Why relevant**: State-of-the-art graph-based multimodal fusion

### 5. Depression Detection Through Multi-Modal Data
- **Repository**: https://github.com/notmanan/Depression-Detection-Through-Multi-Modal-Data
- **Purpose**: Multi-modal depression detection
- **Why relevant**: Additional DAIC-WOZ processing code

---

## Integration with Research Project

### For Multimodal Data Alignment
These repositories demonstrate various approaches to multimodal fusion:
1. **Early fusion**: Concatenating features from different modalities
2. **Late fusion**: Combining predictions from modality-specific models
3. **Hybrid fusion**: Hierarchical or attention-based integration
4. **Contrastive learning**: Aligning representations across modalities

### For Adolescent Mental Health
While most repositories use adult datasets (DAIC-WOZ), the techniques are applicable to adolescent populations when combined with appropriate datasets (e.g., from digital phenotyping studies).

### Baseline Models
These repositories provide strong baselines for:
- Depression detection accuracy
- Multimodal fusion strategies
- Feature extraction pipelines
- Evaluation methodologies

---

## Recommended Workflow

1. **Study existing implementations**:
   - Review code structure and architecture
   - Understand data preprocessing steps
   - Examine fusion strategies

2. **Adapt for adolescent data**:
   - Modify for smartphone sensor data
   - Incorporate active + passive features
   - Adjust for shorter time windows

3. **Implement novel approaches**:
   - Add contrastive learning pretraining
   - Experiment with transformer architectures
   - Develop graph-based fusion models

4. **Evaluate systematically**:
   - Use consistent evaluation metrics
   - Compare against baselines
   - Report cross-validation results

---

## Common Dependencies

Most repositories require:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Common packages
pip install numpy pandas scikit-learn
pip install torch torchvision torchaudio  # PyTorch
pip install tensorflow  # Or TensorFlow
pip install librosa soundfile  # Audio processing
pip install opencv-python  # Video processing
pip install transformers  # For BERT, etc.
pip install datasets  # HuggingFace datasets
```

---

## Notes on Usage

- **Dataset Access**: Most code requires DAIC-WOZ dataset - follow instructions in datasets/README.md
- **Computational Requirements**: Deep learning models may require GPU
- **Reproducibility**: Check for random seeds and version specifications
- **Licenses**: Respect repository licenses and cite original papers

---

## Citation

If you use code from these repositories, please cite the original papers:

**Automatic Depression Detector**:
```
@misc{automatic-depression-detector,
  author = {Derong Chen},
  title = {Automatic Depression Detection},
  year = {2019},
  publisher = {GitHub},
  url = {https://github.com/derong97/automatic-depression-detector}
}
```

**Multimodal Depression from Video**:
```
@article{cosma2024reading,
  title={Reading Between the Frames: Multi-Modal Non-Verbal Depression Detection in Videos},
  author={Cosma, Adrian},
  journal={arXiv preprint},
  year={2024}
}
```

**Context-Aware Multi-Modal Depression Detection**:
```
@inproceedings{ghandeharioun2019context,
  title={Context Aware Deep Learning for Multi Modal Depression Detection},
  author={Ghandeharioun, Asma and others},
  booktitle={ICASSP 2019},
  year={2019}
}
```

---

## Future Directions

Potential enhancements to explore:
1. **Contrastive pretraining**: Like in digital phenotyping paper
2. **Graph neural networks**: For modeling relationships between modalities
3. **Transformer architectures**: For better temporal modeling
4. **Transfer learning**: From adult to adolescent populations
5. **Federated learning**: For privacy-preserving multimodal analysis

---

## Contact & Support

For questions about specific repositories, please open issues on the respective GitHub pages or contact the original authors.
