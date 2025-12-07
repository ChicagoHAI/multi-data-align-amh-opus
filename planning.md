# Research Planning Document

**Research Title**: Comprehensive Review of Multimodal Data Alignment Techniques for Adolescent Mental Health AI Modeling

**Research Domain**: Machine Learning for Healthcare

**Date**: December 7, 2025

---

## Research Question

**Primary Question**: Does effective alignment and integration of multimodal data improve the accuracy and interpretability of AI models for adolescent mental health outcomes?

**Specific Sub-Questions**:
1. How do different fusion strategies (early, late, cross-attention) impact model performance on multimodal mental health data?
2. What is the quantitative improvement from explicit alignment methods compared to naive concatenation?
3. Can we identify which modalities contribute most to prediction accuracy through interpretability analysis?
4. How robust are multimodal alignment techniques across different conditions (stress vs. baseline)?

---

## Background and Motivation

### Why This Is Important

1. **Clinical Urgency**: 75% of mental disorders emerge during adolescence, making early detection critical for prevention and intervention

2. **Data Heterogeneity Challenge**: Mental health assessment involves diverse data types:
   - Self-reported questionnaires and ecological momentary assessments
   - Physiological signals (ECG, EDA, respiration)
   - Speech, language, and text
   - Facial expressions and video
   - Smartphone sensor data

3. **Multimodal Advantage**: Literature shows that combining modalities captures richer signatures of psychiatric conditions than single modalities alone

4. **Alignment Gap**: Most approaches use naive feature concatenation; systematic comparison of alignment techniques is lacking

### What Gap Does This Fill?

From the literature review:
- **Methodological Gap**: Limited exploration of explicit alignment methods (contrastive learning, cross-attention) specifically for mental health data
- **Evaluation Gap**: No systematic comparison of fusion strategies on identical data with standardized evaluation
- **Interpretability Gap**: Black-box models lack clinically actionable explanations

**Expected Impact**: Provide evidence-based guidance on which alignment techniques work best for multimodal mental health data, improving both accuracy and clinical interpretability.

---

## Hypothesis Decomposition

### Main Hypothesis
"Effective alignment and integration of multimodal data improve the accuracy and interpretability of AI models for adolescent mental health outcomes."

### Decomposed Testable Components

**H1: Performance Hypothesis**
- **H1a**: Multimodal models outperform unimodal baselines
- **H1b**: Explicit alignment methods (cross-attention, contrastive learning) outperform naive concatenation
- **H1c**: The performance gain is statistically significant (p < 0.05) with meaningful effect size (Cohen's d > 0.5)

**H2: Robustness Hypothesis**
- **H2a**: Aligned multimodal representations generalize better to new subjects (LOSO cross-validation)
- **H2b**: Alignment methods are robust across different task conditions (stress, baseline, amusement, meditation)

**H3: Interpretability Hypothesis**
- **H3a**: Attention weights from cross-attention models reveal clinically meaningful modality interactions
- **H3b**: Feature importance analysis shows complementary contributions from different modalities

### Success Criteria

**Minimum Success**: Demonstrate that at least one multimodal alignment method significantly outperforms unimodal baselines on balanced accuracy

**Full Success**:
- Show >5 percentage point improvement in balanced accuracy with multimodal alignment
- Demonstrate statistical significance (p < 0.05)
- Provide interpretable explanations of which modalities contribute to decisions
- Show consistent results across multiple evaluation metrics

---

## Proposed Methodology

### Approach

Given the available resources and research constraints, I will:

1. **Use WESAD Dataset** (publicly available, physiological modalities)
   - 15 participants with wearable stress and affect detection data
   - Multiple physiological modalities: ECG, EDA, EMG, Respiration, Temperature, Acceleration
   - Task: Stress detection (baseline vs. stress vs. amusement vs. meditation)
   - **Rationale**: While not adolescent-specific, WESAD provides clean multimodal data to rigorously test alignment techniques that will generalize to adolescent populations

2. **Systematic Comparison** of alignment strategies:
   - Unimodal baselines (each modality independently)
   - Early fusion (naive concatenation)
   - Late fusion (ensemble voting)
   - Cross-attention fusion (explicit alignment)
   - **Rationale**: Cover the spectrum from simplest to most sophisticated alignment

3. **Rigorous Evaluation**:
   - Leave-One-Subject-Out (LOSO) cross-validation for generalization
   - Multiple metrics: Balanced accuracy, F1-score, confusion matrices
   - Statistical significance testing
   - **Rationale**: LOSO tests real-world generalization to new individuals

4. **Interpretability Analysis**:
   - Feature importance (SHAP values or permutation importance)
   - Attention weight visualization
   - Per-modality contribution analysis
   - **Rationale**: Align with clinical needs for explainable predictions

### Experimental Steps

#### Step 1: Data Preparation (Est. 30 min)
1. Download WESAD dataset from UCI repository
2. Load and validate all 15 subjects' data
3. Extract key physiological modalities:
   - **Chest sensors**: ECG, EDA, EMG, Respiration, Temperature
   - **Wrist sensors**: BVP, EDA (secondary)
   - **Focus on chest sensors** for consistency
4. Preprocess signals:
   - Normalize to zero mean, unit variance
   - Segment into windows (e.g., 30-second segments)
   - Extract statistical features (mean, std, min, max, median, IQR)
5. Create labels: 4-class (baseline, stress, amusement, meditation) or binary (stress vs. non-stress)
6. Split data: Leave-One-Subject-Out strategy
7. **Validation**: Check class distributions, ensure no data leakage

**Rationale**:
- Window-based feature extraction is standard for physiological signals
- Statistical features capture temporal dynamics while keeping computational cost low
- Binary task simplifies interpretation; 4-class tests robustness

#### Step 2: Unimodal Baselines (Est. 30 min)
Implement separate models for each modality:

1. **ECG-only**: Logistic Regression, Random Forest, MLP
2. **EDA-only**: Same models
3. **EMG-only**: Same models
4. **Respiration-only**: Same models
5. **Temperature-only**: Same models

**Architecture**: Start with Logistic Regression and Random Forest for interpretability, add MLP if time permits

**Evaluation**: LOSO cross-validation, track balanced accuracy, F1-score

**Rationale**:
- Establishes performance floor
- Identifies which individual modalities are most predictive
- Validates evaluation pipeline before multimodal experiments

#### Step 3: Early Fusion Baseline (Est. 20 min)
1. Concatenate all modality features into single vector
2. Train same models as unimodal (Logistic Regression, Random Forest, MLP)
3. Evaluate with LOSO cross-validation

**Rationale**:
- Simple multimodal baseline
- Standard approach in many papers
- Should improve over unimodal but lacks explicit alignment

#### Step 4: Late Fusion Baseline (Est. 20 min)
1. Train separate model for each modality (from Step 2)
2. Combine predictions via:
   - **Majority voting** (for classification)
   - **Average probabilities** (soft voting)
3. Evaluate with LOSO cross-validation

**Rationale**:
- Robust to modality-specific noise
- Doesn't model cross-modal interactions
- Common ensemble approach

#### Step 5: Cross-Attention Fusion (Est. 60 min)
Implement simplified cross-attention mechanism:

1. **Architecture**:
   - Modality-specific encoders (small MLPs) for each modality
   - Cross-attention layers: each modality attends to others
   - Aggregation: weighted combination based on attention
   - Classification head: final MLP for prediction

2. **Implementation**:
   ```python
   # Pseudo-code
   for each modality m:
       encoded_m = encoder_m(modality_m)

   # Cross-attention
   for each modality m:
       attention_weights = softmax(encoded_m @ encoded_others.T)
       aligned_m = attention_weights @ encoded_others

   combined = concatenate([aligned_1, aligned_2, ..., aligned_M])
   output = classifier(combined)
   ```

3. **Training**: Adam optimizer, cross-entropy loss, early stopping

**Rationale**:
- Explicitly models cross-modal alignment
- Attention weights provide interpretability
- State-of-the-art in multimodal fusion (per literature review)
- Can be implemented in PyTorch relatively quickly

#### Step 6: Ablation Studies (Est. 30 min)
1. **Modality Contribution**: Remove one modality at a time, measure performance drop
2. **Fusion Strategy**: Compare all fusion approaches on identical data
3. **Feature Sets**: Compare statistical features vs. raw signal (if time permits)

**Rationale**: Identifies which components contribute most to performance

---

### Baselines

**Unimodal Baselines** (from literature):
- ECG-only: ~70-75% accuracy for stress detection
- EDA-only: ~72-78% accuracy
- Respiration-only: ~65-70% accuracy

**Multimodal Baselines** (from WESAD paper and literature):
- Early Fusion (SVM, RF): 80-85% accuracy
- CNN-LSTM: 90-95% accuracy
- Wavelet + DenseNet-LSTM: 97-99% accuracy (state-of-the-art)

**Our Targets**:
- Achieve >80% balanced accuracy with early fusion
- Achieve >85% balanced accuracy with cross-attention
- Demonstrate statistical improvement over unimodal baselines

---

### Evaluation Metrics

#### Primary Metrics
1. **Balanced Accuracy**: Average recall per class
   - **Why**: WESAD has some class imbalance; balanced accuracy handles this
   - **Interpretation**: Overall performance across all classes equally

2. **F1 Score (Macro)**: Harmonic mean of precision and recall, averaged across classes
   - **Why**: Complements balanced accuracy, penalizes low precision
   - **Interpretation**: Overall classification quality

#### Secondary Metrics
3. **Confusion Matrix**: Detailed breakdown of predictions
   - **Why**: Shows which classes are confused with which
   - **Interpretation**: Error pattern analysis

4. **AUC-ROC**: For binary stress vs. non-stress
   - **Why**: Threshold-independent performance measure
   - **Interpretation**: Discriminative ability

5. **Per-Class Accuracy**: Precision, recall, F1 for each class
   - **Why**: Identifies class-specific performance
   - **Interpretation**: Clinical relevance (e.g., stress detection sensitivity)

#### Interpretability Metrics
6. **Feature Importance**: Top-k most important features per modality
   - **Method**: Permutation importance or SHAP values
   - **Why**: Clinical interpretability

7. **Attention Weights**: For cross-attention model
   - **Visualization**: Heatmap of modality-to-modality attention
   - **Why**: Shows which modalities interact most

---

### Statistical Analysis Plan

#### Hypothesis Testing

**H1: Multimodal > Unimodal**
- **Test**: Paired t-test comparing multimodal vs. best unimodal across 15 subjects (LOSO folds)
- **Null hypothesis**: No difference in balanced accuracy
- **Alternative**: Multimodal has higher balanced accuracy
- **Significance level**: α = 0.05
- **Effect size**: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)

**H2: Aligned > Naive Concatenation**
- **Test**: Paired t-test comparing cross-attention vs. early fusion across 15 subjects
- **Null hypothesis**: No difference
- **Alternative**: Cross-attention has higher balanced accuracy
- **Significance level**: α = 0.05

**H3: Late Fusion vs. Early Fusion**
- **Test**: Paired t-test
- **Exploratory**: Understand when ensemble methods help

#### Robustness Checks
1. **Cross-Validation Stability**: Report mean ± std across all LOSO folds
2. **Sensitivity to Random Seed**: Run experiments with 3 different seeds
3. **Class-Wise Performance**: Ensure no class has dramatically low performance

#### Multiple Comparison Correction
- If testing >3 pairwise comparisons, apply Bonferroni correction
- Report both corrected and uncorrected p-values

---

## Expected Outcomes

### If Hypothesis Supported

**Evidence for H1 (Multimodal > Unimodal)**:
- Balanced accuracy: Multimodal 85-90% vs. best unimodal 75-80%
- Statistical significance: p < 0.05 with medium-to-large effect size
- **Interpretation**: Confirms that combining modalities captures complementary information

**Evidence for H2 (Aligned > Naive)**:
- Balanced accuracy: Cross-attention 87-90% vs. early fusion 82-85%
- Statistical significance: p < 0.05
- **Interpretation**: Explicit alignment improves over naive concatenation

**Evidence for H3 (Interpretability)**:
- Attention weights show ECG ↔ EDA interaction (known correlation in stress response)
- Feature importance reveals modality-specific contributions
- **Interpretation**: Models align with physiological understanding of stress

### If Hypothesis Refuted

**Scenario 1: Multimodal ≈ Unimodal**
- **Possible reasons**:
  - One modality dominates (e.g., EDA alone is sufficient)
  - Modalities are redundant (highly correlated)
  - Dataset too small for multimodal advantage
- **Next steps**: Analyze feature correlations, test on larger dataset

**Scenario 2: Aligned ≈ Naive**
- **Possible reasons**:
  - Dataset is too clean (little noise for alignment to help)
  - Modalities are naturally aligned (temporal synchrony)
  - Alignment method not well-suited for this data type
- **Next steps**: Test on noisier dataset, try different alignment techniques

**Scenario 3: Poor Overall Performance (<70% accuracy)**
- **Possible reasons**:
  - Feature extraction inadequate
  - Model complexity insufficient
  - Task is inherently difficult
- **Next steps**: Try raw signal processing, deeper networks, consult domain experts

**Scientific Value**: Negative results are valuable! They would inform:
- When simple methods suffice (practical guidance)
- Limitations of current alignment techniques
- Need for better datasets or more sophisticated approaches

---

## Timeline and Milestones

### Phase 1: Data Preparation and EDA (30-45 min)
- [ ] Download and load WESAD dataset
- [ ] Verify data integrity (15 subjects, all modalities present)
- [ ] Exploratory data analysis:
  - Signal quality checks
  - Class distributions
  - Modality correlations
- [ ] Feature extraction pipeline
- [ ] LOSO split preparation
- **Milestone**: Dataset loaded, features extracted, ready for modeling

### Phase 2: Baseline Implementations (60-75 min)
- [ ] Unimodal baselines (5 modalities × 2-3 models)
- [ ] Early fusion baseline
- [ ] Late fusion baseline
- [ ] Validation: Check that results are reasonable
- **Milestone**: All baselines running, results logged

### Phase 3: Cross-Attention Implementation (60-90 min)
- [ ] Design architecture
- [ ] Implement in PyTorch
- [ ] Training loop with early stopping
- [ ] Hyperparameter tuning (learning rate, hidden dims)
- [ ] LOSO evaluation
- **Milestone**: Cross-attention model trained and evaluated

### Phase 4: Analysis and Evaluation (45-60 min)
- [ ] Compile all results
- [ ] Statistical significance testing
- [ ] Confusion matrices and visualizations
- [ ] Interpretability analysis (attention weights, feature importance)
- [ ] Ablation studies
- **Milestone**: Complete analysis with statistical tests and figures

### Phase 5: Documentation (30-45 min)
- [ ] Create comprehensive REPORT.md
- [ ] Create README.md
- [ ] Ensure reproducibility (save all configs, seeds)
- [ ] Clean up code with comments
- **Milestone**: Complete documentation ready for review

**Total Estimated Time**: 4-6 hours
**Buffer**: 20-30% for debugging, unexpected issues

---

## Potential Challenges

### Challenge 1: Dataset Download
**Issue**: WESAD download might be slow or unavailable
**Mitigation**:
- Try multiple download sources (UCI, Kaggle, direct link)
- If unavailable, use synthetic multimodal data to demonstrate methodology
**Contingency**: Document methodology even if dataset unavailable; scientific value in approach

### Challenge 2: Computational Resources
**Issue**: Training deep models might be slow on CPU
**Mitigation**:
- Start with simpler models (Logistic Regression, Random Forest)
- Use small network for cross-attention (2-3 layers, hidden dim 64-128)
- Reduce training epochs, use early stopping
**Contingency**: Stick to classical ML if deep learning too slow

### Challenge 3: Poor Baseline Performance
**Issue**: Models perform poorly (<60% accuracy)
**Mitigation**:
- Check for data leakage or preprocessing errors
- Try different feature extraction (Fourier transform, wavelets)
- Reduce task complexity (4-class → binary)
**Contingency**: Focus on relative comparisons even if absolute performance is modest

### Challenge 4: No Significant Difference
**Issue**: Multimodal methods don't significantly outperform unimodal
**Mitigation**:
- Analyze feature correlations (might be redundant)
- Try different modality combinations
- Report negative result honestly with analysis of why
**Contingency**: Negative results are scientifically valid; document thoroughly

### Challenge 5: Implementation Bugs
**Issue**: Cross-attention implementation has bugs
**Mitigation**:
- Start simple: implement without attention first
- Unit test each component
- Validate on toy data before full dataset
- Use existing PyTorch attention modules
**Contingency**: Fall back to simpler hybrid fusion if attention too complex

---

## Success Criteria

### Minimum Viable Research

**Must Have**:
1. ✅ Unimodal baselines for at least 3 modalities
2. ✅ One multimodal baseline (early or late fusion)
3. ✅ LOSO cross-validation evaluation
4. ✅ Statistical comparison (t-test or equivalent)
5. ✅ REPORT.md documenting results and methodology
6. ✅ Reproducible code with clear documentation

**Scientific Value**: Even minimal scope demonstrates rigorous methodology and provides actionable insights

### Full Success

**Should Have**:
1. ✅ All unimodal baselines (5+ modalities)
2. ✅ Multiple multimodal approaches (early, late, cross-attention)
3. ✅ Comprehensive statistical analysis with effect sizes
4. ✅ Interpretability analysis (attention weights, feature importance)
5. ✅ Ablation studies
6. ✅ Multiple visualizations (confusion matrices, attention heatmaps, performance plots)

**Scientific Value**: Comprehensive comparison provides clear guidance on best practices

### Stretch Goals

**Nice to Have** (if time permits):
1. Contrastive learning pretraining (as in digital phenotyping paper)
2. Graph-based fusion (modality graph)
3. Raw signal processing with CNN
4. Additional datasets (synthetic or from HuggingFace)
5. Fairness analysis (demographic stratification if metadata available)

---

## Risk Mitigation Summary

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset unavailable | Low | High | Multiple download sources; synthetic data fallback |
| Slow computation | Medium | Medium | Start simple; use classical ML; small networks |
| Poor performance | Medium | Low | Focus on relative comparisons; reduce task complexity |
| No significant difference | Low | Medium | Analyze correlations; report negative result honestly |
| Implementation bugs | Medium | Medium | Incremental development; unit tests; simpler fallback |
| Time overrun | Medium | Low | Prioritize core experiments; clear stopping criteria |

---

## Key Decisions and Rationale

### Decision 1: Use WESAD Instead of DAIC-WOZ
**Rationale**:
- WESAD is publicly available (no approval process)
- Multiple physiological modalities suitable for alignment study
- While not adolescent-specific, methodology generalizes
- Literature review shows physiological signals are effective for mental health

**Trade-off**: Not adolescent-specific, but enables rigorous methodology development

### Decision 2: Focus on Classical ML + Simple Deep Learning
**Rationale**:
- Faster to implement and train
- More interpretable (Logistic Regression, Random Forest)
- Sufficient to test alignment hypothesis
- Cross-attention adds explicit alignment without excessive complexity

**Trade-off**: Won't achieve state-of-the-art performance, but tests hypothesis rigorously

### Decision 3: Statistical Features vs. Raw Signals
**Rationale**:
- Statistical features (mean, std, etc.) are standard in literature
- Much faster to compute and model
- Interpretable for clinicians
- Raw signal processing requires CNNs/RNNs (more complex)

**Trade-off**: May miss temporal dynamics, but enables thorough comparison of fusion strategies

### Decision 4: LOSO Cross-Validation
**Rationale**:
- Tests generalization to new individuals (critical for real-world deployment)
- Standard in mental health ML research
- 15 subjects provides 15 folds (reasonable sample size)

**Trade-off**: More computationally expensive than k-fold, but scientifically rigorous

---

## Alignment with Literature Review

### Key Insights from Literature

1. **Fusion Strategies** (Al Sahili et al., 2024):
   - Cross-attention is dominant for aligned modalities
   - Early fusion suits clean, synchronized signals
   - Hybrid approaches balance complexity and performance

2. **Digital Phenotyping** (Kadirvelu et al., 2025):
   - Multimodal (active + passive) outperforms unimodal by 4-10 pp
   - Contrastive pretraining improves performance (p<0.001)
   - SHAP reveals complementary modality contributions

3. **Baselines** (from survey):
   - WESAD: Traditional ML 80-85%, CNN-LSTM 90-95%, Wavelet+DenseNet 97-99%
   - DAIC-WOZ: Cross-attention transformer 82-88% F1

### How Our Experiment Leverages This

1. **Direct Test of Fusion Strategies**: Systematic comparison on identical data
2. **Replication**: Aim to replicate multimodal advantage (4-10 pp improvement)
3. **Extension**: Add cross-attention not tested on WESAD in original paper
4. **Interpretability**: Follow digital phenotyping paper's use of SHAP/feature importance

---

## Documentation Strategy

### During Experiments
- **Jupyter Notebook**: Interactive exploration, EDA, experiments
- **Logging**: Track all hyperparameters, results, random seeds
- **Version Control**: Git commits at each milestone

### Final Documentation
- **REPORT.md**: Comprehensive research report (Executive Summary, Methodology, Results, Analysis, Conclusions)
- **README.md**: Quick overview, key findings, reproduction instructions
- **Code Comments**: Inline documentation explaining complex logic
- **requirements.txt**: Reproducible environment

---

## Ethical Considerations

1. **Data Use**: Comply with WESAD's academic/non-commercial license
2. **Reporting**: Honest reporting of all results (positive and negative)
3. **Limitations**: Acknowledge that WESAD is not adolescent-specific
4. **Generalizability**: Discuss applicability to adolescent populations
5. **Clinical Translation**: Frame as research prototype, not clinical tool

---

## Next Steps After Planning

1. **Download WESAD dataset** immediately (might take time)
2. **Set up Jupyter notebook** for interactive development
3. **Start with EDA** to understand data characteristics
4. **Implement unimodal baselines** first to validate pipeline
5. **Incrementally add complexity** (early fusion → late fusion → cross-attention)
6. **Document as you go** to avoid forgetting decisions

---

## Summary

This research plan provides a **systematic, rigorous investigation** of multimodal data alignment techniques for mental health AI modeling. By using the publicly available WESAD dataset, we can test our hypothesis that explicit alignment methods improve both accuracy and interpretability compared to naive baselines.

**Key Strengths**:
- ✅ Leverages comprehensive literature review
- ✅ Uses publicly available dataset (no approval delays)
- ✅ Systematic comparison of fusion strategies
- ✅ Rigorous evaluation (LOSO, statistical tests)
- ✅ Focus on interpretability (clinical relevance)
- ✅ Clear success criteria and contingency plans

**Key Limitations**:
- ⚠️ Not adolescent-specific (methodology will generalize)
- ⚠️ Moderate sample size (15 subjects)
- ⚠️ Lab-based data (not ecologically valid)

Despite limitations, this experiment will provide **actionable insights** on multimodal alignment techniques with clear implications for adolescent mental health AI systems.

**Decision Point**: Proceed with implementation ✅

---

**Planning Phase Complete**: December 7, 2025
**Next Phase**: Implementation (Data Preparation)
