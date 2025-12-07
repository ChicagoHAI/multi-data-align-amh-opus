# Literature Review: Multimodal Data Alignment for Adolescent Mental Health AI Modeling

**Research Hypothesis**: Effective alignment and integration of multimodal data improve the accuracy and interpretability of AI models for adolescent mental health outcomes.

**Research Domain**: Machine Learning for Healthcare

**Date**: December 7, 2025

---

## 1. Research Area Overview

Multimodal machine learning (MML) has emerged as a transformative approach for mental health assessment, particularly for adolescent populations. The convergence of diverse data streams—including speech, text, facial expressions, physiological signals, and smartphone sensor data—enables more comprehensive and ecologically valid characterization of psychiatric conditions than any single modality alone.

### Key Trends

1. **Shift from Unimodal to Multimodal**: Early studies relied on isolated data streams, but recent research demonstrates that integrating heterogeneous modalities captures richer, more complex signatures of mental health conditions.

2. **Digital Phenotyping**: Smartphones enable continuous, unobtrusive data collection combining active self-reports with passive sensor measurements, offering scalable solutions for adolescent mental health monitoring.

3. **Advanced Fusion Architectures**: Evolution from simple concatenation to sophisticated transformer-based, graph-based, and hybrid fusion strategies that explicitly model cross-modal alignment.

4. **Focus on Adolescents**: Growing recognition that early identification and intervention during adolescence (when 75% of mental disorders emerge) is critical for prevention.

---

## 2. Key Papers

### Paper 1: Multimodal Machine Learning in Mental Health: A Survey (2024)

**Citation**: Al Sahili, Z., Patras, I., & Purver, M. (2024). Multimodal Machine Learning in Mental Health: A Survey of Data, Algorithms, and Challenges. *arXiv preprint arXiv:2407.16804*.

**Key Contribution**: First comprehensive, clinically grounded synthesis of MML for mental health, cataloging 26 public datasets and systematically comparing 28 models across transformer, graph, and hybrid fusion strategies.

**Methodology**:
- Systematic review of multimodal approaches
- Taxonomy of data modalities (text, audio, video, physiological)
- Comparative analysis of fusion strategies (early, late, hybrid)
- Framework for understanding representation learning and cross-modal alignment

**Datasets Cataloged**: DAIC-WOZ, E-DAIC, AVEC 2013, D-Vlog, WESAD, DEAP, MuSE, and 19 others spanning depression, stress, PTSD, bipolar disorder, and emotion recognition.

**Key Findings**:
- **Fusion Strategies**: Cross-attention has emerged as dominant mechanism for aligned modalities; early latent fusion suits weak synchrony
- **Pre-training Matters**: Domain-specific pre-training (MentalBERT, wav2vec 2.0) systematically boosts downstream accuracy
- **Performance Gains**: Transformer toolkit advances state-of-the-art by 2-10 percentage points across domains
- **Hybrid CNN/RNN**: Still provides strong, computationally efficient baselines with favorable speed-accuracy trade-offs

**Methodological Insights**:
1. *When fusion occurs matters*: Early concatenation for synchronous/clean signals; late voting for heavy noise
2. *Attention mechanisms* consistently boost performance by spotlighting reliable channels
3. *Graph neural networks* excel at encoding heterogeneous entities (brain regions, interview segments) as nodes with typed edges

**Relevance to Our Research**:
- Provides comprehensive framework for understanding multimodal fusion strategies
- Identifies key datasets (DAIC-WOZ, WESAD) applicable to our research
- Highlights importance of cross-modal alignment—directly relevant to our hypothesis
- Documents state-of-the-art baselines for comparison

---

### Paper 2: Digital Phenotyping for Adolescent Mental Health (2025)

**Citation**: Kadirvelu, B., Bellido Bel, T., Freccero, A., Di Simplicio, M., Nicholls, D., & Faisal, A.A. (2025). Digital Phenotyping for Adolescent Mental Health: A Feasibility Study Employing Machine Learning to Predict Mental Health Risk From Active and Passive Smartphone Data. *arXiv preprint arXiv:2501.08851*.

**Key Contribution**: First study to integrate active (self-reported) and passive (sensor-based) smartphone data using contrastive learning for predicting multiple adolescent mental health outcomes in non-clinical populations.

**Methodology**:
- **Population**: 103 adolescents (mean age 16.1 years) from London schools
- **Duration**: 14-day monitoring using Mindcraft app
- **Active Data**: Daily self-reports (mood, sleep quality, loneliness, negative thinking, etc.) on 1-7 scale
- **Passive Data**: 92 engineered features from 8 sensor categories (location, step count, app usage, ambient noise, battery, screen brightness, Mindcraft usage)
- **Novel ML Approach**: Contrastive learning with triplet margin loss for user-specific feature stabilization
- **Evaluation**: Leave-one-subject-out cross-validation

**Mental Health Outcomes Predicted**:
1. SDQ High-risk (Strengths and Difficulties Questionnaire ≥16)
2. Insomnia (Sleep Condition Indicator <17)
3. Suicidal ideation (PHQ-9 item ≥1)
4. Eating disorders (ED-15 >2.69)

**Results**:
- **Balanced Accuracies**: 0.71 (SDQ), 0.67 (insomnia), 0.77 (suicidal ideation), 0.70 (eating disorders)
- **Multimodal Advantage**: Combined active+passive outperformed either alone
- **Contrastive Learning Benefit**: Pretraining improved performance (p<0.001) over non-pretrained models
- **Key Predictors**:
  - Active: Negative thinking, racing thoughts, self-care, hopefulness, loneliness
  - Passive: Location entropy, ambient light, step count, latitude variability

**Methodological Innovation**:
- **Contrastive Pretraining**: Triplet margin loss clusters user-specific features across days, reducing day-to-day variability
- **User-Level Aggregation**: Day-wise predictions averaged for single user-level prediction
- **SHAP Interpretability**: Feature importance analysis reveals clinically meaningful predictors

**Relevance to Our Research**:
- **Directly Addresses Hypothesis**: Demonstrates that multimodal data alignment (active + passive) improves prediction accuracy
- **Adolescent Focus**: Non-clinical population matching our research needs
- **Novel Alignment Technique**: Contrastive learning provides concrete approach for data alignment
- **Multiple Outcomes**: Covers diverse mental health conditions (internalizing, externalizing, eating disorders, sleep, suicidality)
- **Practical Framework**: Smartphone-based approach is scalable and ecologically valid

---

### Paper 3: Depression Detection with Multi-Modal Feature Fusion Using Cross-Attention (2024)

**Citation**: Referenced from arXiv:2407.12825 (MFFNC - Multimodal Feature Fusion Network based on Cross-attention)

**Key Contribution**: Proposes MFFNC architecture demonstrating exceptional performance in depression identification through cross-attention-based multimodal fusion.

**Methodology**:
- Cross-attention mechanism for multimodal feature fusion
- Decomposition into modality-specific and modality-common parts
- Recombination via transformer layers

**Relevance**: Demonstrates state-of-the-art cross-attention approach for aligning multimodal mental health data.

---

### Paper 4: Robust Multimodal Representation with Adaptive Experts and Alignment (2025)

**Citation**: arXiv:2503.09498 (MoSARe framework)

**Key Contribution**: Unified approach for robust multimodal representation learning with adaptive experts, specifically handling incomplete multimodal data.

**Key Features**:
- Mixture of experts architecture
- Symmetric aligning mechanisms
- Reconstruction-based learning
- Handles missing modalities gracefully

**Relevance**: Critical for real-world scenarios where not all modalities may be available for all users (e.g., some disable certain sensors).

---

### Paper 5: Cross-Modal Alignment via Variational Copula Modelling (2024)

**Citation**: arXiv:2511.03196

**Key Contribution**: Novel copula-driven multimodal learning framework for learning joint distributions of modalities to capture complex interactions.

**Methodology**:
- Variational copula models for cross-modal dependencies
- Captures non-linear relationships between modalities
- Tested on MIMIC healthcare datasets

**Key Finding**: Superior performance over competitors on public MIMIC datasets.

**Relevance**: Provides statistical framework for understanding cross-modal dependencies—directly relevant to data alignment hypothesis.

---

### Paper 6: Comprehensive Review of Datasets for Clinical Mental Health AI (2024)

**Citation**: arXiv:2508.09809

**Key Contribution**: Systematic catalog of datasets for clinical mental health AI systems.

**Relevance**: Essential reference for identifying validated datasets and understanding dataset characteristics, limitations, and access requirements.

---

### Paper 7: Additional Relevant Work from Survey

**CNN-LSTM Multimodal Model** (0.946 accuracy on depression/anxiety):
- Fuses scale scores and video information
- Demonstrates high diagnostic performance for initial screening

**Voice + Text Fusion with BERT and Wav2vec2.0**:
- Multi-scale convolutional kernels + Bi-LSTM
- Significantly improved over single-modality models

**Brain Network Analysis** (Graph-based):
- Multi-atlas dynamic functional connectivity
- 81% accuracy for insomnia disorder
- Highlights dysfunctional DMN regions

---

## 3. Common Methodologies

### 3.1 Multimodal Fusion Strategies

**Early Fusion**:
- Concatenate features from all modalities before model
- **Pros**: Simple, allows learning joint representations
- **Cons**: Requires synchronized data, sensitive to noise
- **Example**: Concat(text_features, audio_features, video_features) → Classifier

**Late Fusion**:
- Train separate models per modality, combine predictions
- **Pros**: Robust to missing modalities, modality-specific optimization
- **Cons**: Doesn't capture cross-modal interactions
- **Example**: Vote(text_model, audio_model, video_model)

**Hybrid Fusion**:
- Combines early and late fusion strategies
- **Pros**: Captures both modality-specific and cross-modal patterns
- **Cons**: More complex architecture
- **Examples**:
  - Hierarchical attention across modalities
  - Separate encoders + cross-modal attention layers

**Cross-Attention Mechanisms**:
- Queries from one modality attend to keys/values from another
- **Implementation**: Multi-head cross-attention transformers
- **Benefit**: Explicitly models cross-modal alignment
- **Usage**: Dominant for temporally aligned modalities

**Graph-Based Fusion**:
- Model modalities and their relationships as graph
- Nodes represent modality features or segments
- Edges represent relationships (temporal, semantic, phenotypic)
- **Examples**:
  - Heterogeneous graph transformers
  - Graph convolutional networks with typed edges
  - Population graphs for multi-subject learning

### 3.2 Contrastive Learning for Alignment

**Triplet Margin Loss** (from digital phenotyping paper):
```
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```
- Anchor: User on day i
- Positive: Same user on day j
- Negative: Different user
- **Effect**: Clusters user-specific features, reduces day-to-day variability

**Other Contrastive Approaches**:
- SimCLR-style for multimodal representations
- Contrastive predictive coding for temporal sequences
- CLIP-style for vision-language alignment

### 3.3 Deep Learning Architectures

**Transformers**:
- Multi-head self-attention for intra-modal processing
- Cross-attention for inter-modal fusion
- Positional encodings for temporal data (time2vec)
- **Variants**: BERT, RoBERTa, ViT, wav2vec 2.0, Whisper

**Recurrent Networks**:
- LSTM/BiLSTM for sequential data
- GRU for computational efficiency
- **Usage**: Temporal modeling of smartphone sensor data, interview transcripts

**Convolutional Networks**:
- CNNs for spatial features (images, spectrograms)
- Temporal convolutions for time-series
- **Usage**: Facial features, audio spectrograms

**Graph Neural Networks**:
- GCN, GAT for graph-structured data
- Spectral vs. spatial convolutions
- **Usage**: Brain networks, social graphs, modality relationships

---

## 4. Standard Baselines

### Depression Detection (DAIC-WOZ)

| Method | Modalities | F1 Score | Balanced Accuracy |
|--------|-----------|----------|-------------------|
| Audio-only LSTM | A | 0.60-0.70 | - |
| Text-only BERT | T | 0.65-0.75 | - |
| Video-only CNN | V | 0.55-0.65 | - |
| Early Fusion CNN+LSTM | A+V+T | 0.75-0.81 | - |
| Cross-Attention Transformer | A+V+T | 0.82-0.88 | 0.70-0.75 |
| Graph Attention Network | A+V+T | 0.90-0.95 | 0.80-0.85 |

### Adolescent Mental Health Prediction (Smartphone Data)

| Method | Data Type | Balanced Accuracy |
|--------|-----------|-------------------|
| Active only | Self-reports | 0.63-0.71 |
| Passive only | Sensors | 0.44-0.66 |
| Combined (no pretraining) | Active+Passive | 0.64-0.69 |
| Combined + Contrastive | Active+Passive | 0.67-0.77 |

### Stress Detection (WESAD)

| Method | Accuracy |
|--------|----------|
| Traditional ML (SVM, RF) | 0.80-0.85 |
| CNN-LSTM | 0.90-0.95 |
| Wavelet + DenseNet-LSTM | 0.97-0.99 |

---

## 5. Evaluation Metrics

### Standard Metrics

**Classification**:
- **Accuracy**: Overall correct predictions
- **Balanced Accuracy**: Average of recall per class (handles class imbalance)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
- **AUC-PR**: Area under precision-recall curve (better for imbalanced data)

**Regression** (for severity scores):
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Pearson r**: Correlation with ground truth

### Domain-Specific Metrics

**Clinical Relevance**:
- **Sensitivity**: True positive rate (critical for mental health screening)
- **Specificity**: True negative rate (avoid over-diagnosis)
- **PPV**: Positive predictive value at low prevalence
- **NPV**: Negative predictive value

**Fairness Metrics**:
- Performance stratified by demographics (age, gender, ethnicity)
- Equality of opportunity across groups
- False negative rate parity (critical for underserved populations)

### Evaluation Protocols

**Cross-Validation**:
- **Leave-One-Subject-Out (LOSO)**: Tests generalization to new individuals
- **K-Fold**: Standard for larger datasets
- **Temporal**: Train on earlier data, test on later (for longitudinal studies)

**Data Splits**:
- Participant-level (not sample-level) to avoid data leakage
- Stratified by outcome to maintain class balance
- Held-out test sets for final evaluation

---

## 6. Datasets in the Literature

### Primary Datasets for Multimodal Mental Health

| Dataset | Modalities | Size | Target | Access |
|---------|-----------|------|--------|--------|
| DAIC-WOZ | A+V+T | 189 | Depression, PTSD | Form required |
| E-DAIC | A+V+T | 275 | Depression, PTSD, Anxiety | Form required |
| AVEC 2013 | V+A | 340 | Depression | Challenge |
| D-Vlog | V+A | YouTube | Depression | Public/Request |
| WESAD | P | 15 | Stress, Affect | Public (UCI) |
| DEAP | V+P | 32 | Emotion | Public |
| MuSE | V+T+P | 28 | Stress | Request |

**Legend**: A=Audio, V=Video, T=Text, P=Physiological

### Datasets by Target Population

**Adults (Clinical)**:
- DAIC-WOZ, E-DAIC, AVEC series
- Focus: Depression, PTSD, anxiety

**Adults (Lab-based)**:
- WESAD, DEAP, MuSE
- Focus: Stress, emotion, affect

**Adolescents** (Limited availability):
- Few dedicated adolescent datasets
- Most research adapts adult methodologies
- Digital phenotyping studies (e.g., Mindcraft) filling gap

### Key Gap: Adolescent Multimodal Datasets

**Current Limitations**:
- Most datasets use adult populations
- Lab-based protocols not ecologically valid for adolescents
- Limited diversity in demographics and cultural representation
- Short-term data collection (single session vs. longitudinal)

**Emerging Solutions**:
- Smartphone-based digital phenotyping (e.g., Mindcraft)
- School-based screening programs
- Naturalistic data collection from social media (ethical concerns)

---

## 7. Gaps and Opportunities

### 7.1 Data Availability and Quality

**Gaps**:
- **Adolescent Focus**: Scarcity of multimodal datasets specifically for adolescent populations
- **Longitudinal Data**: Most datasets provide single-session snapshots, not long-term monitoring
- **Cultural Diversity**: Geographic skew toward Western, university-educated populations
- **Modality Coverage**: Only 25% of datasets combine behavioral channels with physiological signals
- **Label Quality**: Ranges from validated clinical scales to self-report hashtags

**Opportunities**:
- **Digital Phenotyping**: Smartphones enable continuous, ecologically valid data collection
- **Federated Learning**: Allows learning without centralizing sensitive data
- **Synthetic Data**: Augmentation techniques for rare conditions
- **Community-Based Recruitment**: Expand beyond clinical settings to general populations

### 7.2 Methodological Gaps

**Alignment Techniques**:
- **Gap**: Limited exploration of explicit alignment methods (e.g., contrastive learning) in mental health domain
- **Opportunity**: Apply recent advances from vision-language models (CLIP, ALIGN) to mental health multimodal data

**Handling Missing Modalities**:
- **Gap**: Most models require all modalities at inference
- **Opportunity**: Robust fusion methods (e.g., MoSARe) that handle incomplete data

**Interpretability**:
- **Gap**: Black-box models lack clinically actionable explanations
- **Opportunity**:
  - Attention visualization
  - SHAP/LIME for feature importance
  - Counterfactual explanations
  - Align features with DSM-5 symptom clusters

### 7.3 Fairness and Bias

**Gaps**:
- Under-representation of certain demographics in training data
- Sensor biases (e.g., darker skin tones harder to track in video)
- Language variation (AAVE mislabeling)
- Differential access to technology

**Opportunities**:
- Algorithmic debiasing (adversarial training, reweighting)
- Diverse data collection initiatives
- Fairness-aware evaluation (stratified metrics)
- Community-engaged research design

### 7.4 Clinical Translation

**Gaps**:
- **Validation**: Limited real-world clinical validation of ML models
- **Integration**: Challenges integrating into clinical workflows
- **Trust**: Clinicians hesitant to use black-box systems
- **Regulation**: Unclear regulatory pathways for AI-based diagnostics

**Opportunities**:
- **Decision Support**: Frame as assistive tool, not replacement for clinicians
- **Explainability**: Provide interpretable outputs aligned with clinical reasoning
- **Randomized Trials**: Demonstrate clinical utility and cost-effectiveness
- **Ethical Frameworks**: Develop guidelines for responsible AI in mental health

---

## 8. Recommendations for Our Experiment

### 8.1 Recommended Datasets

**Primary**:
1. **DAIC-WOZ** - Gold standard for multimodal depression/PTSD detection
   - Well-established baselines
   - Multimodal (audio, video, text)
   - Clinical validation

2. **WESAD** - Physiological stress detection
   - Publicly available
   - Clean, controlled protocol
   - Complements DAIC-WOZ with physiological modality

**Secondary** (if applicable):
3. **Mental Health Counseling Conversations (HuggingFace)** - Text-based
   - Large scale (100K+ conversations)
   - Pre-training corpus for language models

**Target** (for adolescent focus):
4. **Digital Phenotyping Studies** - Collect own data or collaborate
   - Smartphone-based (active + passive)
   - Adolescent population
   - Ecologically valid

### 8.2 Recommended Baselines

**Unimodal Baselines**:
1. **Audio-only**: LSTM or CNN on spectrograms
2. **Text-only**: BERT/RoBERTa fine-tuned
3. **Video-only**: CNN on facial features or action units

**Multimodal Baselines**:
1. **Early Fusion**: Concatenate features → LSTM/MLP
2. **Late Fusion**: Average or vote predictions
3. **Attention-based**: Cross-attention transformer
4. **Graph-based**: Heterogeneous graph network

**State-of-the-Art**:
1. **With Contrastive Pretraining**: As in digital phenotyping paper
2. **Graph Attention**: Knowledge-aware or heterogeneous GAT
3. **Multimodal Transformer**: LXMERT-style with time encoding

### 8.3 Recommended Metrics

**Primary**:
- **Balanced Accuracy**: Handles class imbalance (common in mental health data)
- **F1 Score (Macro)**: Equal weight to all classes

**Secondary**:
- **AUC-ROC**: Overall discriminative ability
- **AUC-PR**: Performance on minority class (more informative for imbalanced data)
- **Sensitivity/Specificity**: Clinical relevance

**Fairness**:
- **Stratified Performance**: By age, gender, ethnicity (if available)
- **False Negative Rate**: Critical for mental health screening

### 8.4 Recommended Methodological Considerations

**Data Alignment**:
1. **Temporal Alignment**: Ensure modalities are time-synchronized
2. **Feature Alignment**: Normalize scales, distributions across modalities
3. **Semantic Alignment**: Use contrastive learning to align representations

**Model Architecture**:
1. **Modality-Specific Encoders**: Extract rich representations per modality
2. **Cross-Modal Fusion**: Explicit alignment mechanism (attention, graph edges)
3. **Task-Specific Decoder**: Multi-task learning for multiple outcomes

**Training Strategy**:
1. **Pre-training**: On large unlabeled corpus (if available)
2. **Contrastive Learning**: For user-specific or cross-modal alignment
3. **Fine-tuning**: On labeled data with class balancing
4. **Regularization**: Dropout, weight decay to prevent overfitting

**Evaluation**:
1. **LOSO Cross-Validation**: For generalization to new individuals
2. **Ablation Studies**: Compare fusion strategies, modality contributions
3. **Interpretability Analysis**: SHAP values, attention weights
4. **Clinical Validation**: If possible, with domain experts

---

## 9. Methodological Insights

### From Survey Paper

1. **Fusion Strategy Selection**:
   - Use cross-attention for temporally aligned modalities
   - Use early fusion for high-quality, low-noise data
   - Use late fusion or ensemble for heterogeneous, noisy sources
   - Use hybrid approaches for best of both worlds

2. **Architecture Selection**:
   - CNN/RNN: Good baselines, computationally efficient
   - Transformers: State-of-the-art for most tasks, requires more data
   - GNNs: Excellent for capturing relationships, interpretable

3. **Pre-training is Critical**:
   - Domain-specific pre-training (MentalBERT) > general pre-training (BERT)
   - wav2vec 2.0 for audio significantly improves over spectrograms
   - ViT for images requires large datasets or transfer learning

### From Digital Phenotyping Paper

1. **Multimodal Advantage is Real**:
   - Combined active+passive significantly outperforms either alone
   - Effect sizes: 4-10 percentage points improvement in balanced accuracy

2. **Contrastive Learning Works**:
   - User-specific clustering via triplet loss improves stability
   - Reduces day-to-day variability while preserving individual differences
   - Significantly better than no pretraining (p<0.001)

3. **Interpretability Matters**:
   - SHAP reveals clinically meaningful predictors:
     - Active: Negative thinking, racing thoughts, self-care
     - Passive: Location entropy, ambient light, step count
   - Helps build trust with clinicians and users

4. **Practical Considerations**:
   - Active data engagement declines over time (14 → 36 users by day 14)
   - Passive data more sustainable for longitudinal monitoring
   - Not all users enable all sensors → need robust handling

---

## 10. Synthesis and Conclusions

### Key Insights for Multimodal Data Alignment

1. **Alignment Improves Performance**: Across all reviewed papers, explicit modeling of cross-modal relationships (via attention, graphs, or contrastive learning) consistently outperforms naive concatenation.

2. **Multiple Alignment Strategies**:
   - **Temporal**: Synchronize time-series from different sensors
   - **Semantic**: Align representations in shared latent space (contrastive learning)
   - **Relational**: Model dependencies via graphs or cross-attention

3. **Context Matters**:
   - **Data Quality**: Clean, synchronized → early fusion; noisy → late fusion
   - **Task**: Fine-grained (attention), coarse-grained (concatenation)
   - **Population**: Adolescents benefit from longitudinal, ecologically valid data

### Hypothesis Support

**Our hypothesis** states: "Effective alignment and integration of multimodal data improve the accuracy and interpretability of AI models for adolescent mental health outcomes."

**Strong Evidence**:
1. **Accuracy**: Digital phenotyping paper shows 4-10 pp improvement with multimodal fusion
2. **Interpretability**: SHAP analysis reveals complementary information from active and passive data
3. **Alignment Techniques**: Contrastive learning explicitly aligns representations, yielding significant improvements
4. **Adolescent Relevance**: Feasibility demonstrated in non-clinical adolescent population

**Remaining Questions**:
1. Optimal alignment strategy for specific modality combinations?
2. How to handle missing modalities in adolescent populations?
3. Generalization across different adolescent subpopulations?
4. Long-term stability of multimodal predictions?

### Future Directions

1. **Larger Adolescent Datasets**: Community-based recruitment, school partnerships
2. **Advanced Alignment**: Explore copula models, optimal transport for multi-modal alignment
3. **Causal Models**: Move beyond prediction to understanding causal mechanisms
4. **Federated Learning**: Privacy-preserving multimodal learning across institutions
5. **Real-World Deployment**: Integration with school counseling, pediatric care

---

## References

1. Al Sahili, Z., Patras, I., & Purver, M. (2024). Multimodal Machine Learning in Mental Health: A Survey of Data, Algorithms, and Challenges. *arXiv preprint arXiv:2407.16804*.

2. Kadirvelu, B., Bellido Bel, T., Freccero, A., Di Simplicio, M., Nicholls, D., & Faisal, A.A. (2025). Digital Phenotyping for Adolescent Mental Health: A Feasibility Study Employing Machine Learning to Predict Mental Health Risk From Active and Passive Smartphone Data. *arXiv preprint arXiv:2501.08851*.

3. Additional papers cataloged in papers/ directory.

---

**Document Version**: 1.0
**Last Updated**: December 7, 2025
**Total Papers Reviewed**: 7 primary papers + 28 models from survey
