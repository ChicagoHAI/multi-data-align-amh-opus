# Multimodal Mental Health Datasets

This directory contains information and download instructions for datasets used in research on multimodal data alignment techniques for adolescent mental health AI modeling.

**IMPORTANT**: Data files are NOT committed to git due to size and privacy constraints. Follow the download instructions below to obtain the datasets locally.

---

## Dataset 1: DAIC-WOZ (Depression, Anxiety, and PTSD Detection)

### Overview
- **Source**: USC Institute for Creative Technologies
- **Official Website**: https://dcapswoz.ict.usc.edu/
- **Size**: 189 participants, ~3GB total
- **Format**: Audio, video, transcripts, facial features
- **Modalities**: Audio, Video, Text
- **Task**: Depression and PTSD screening
- **Splits**: 189 sessions (7-33 minutes each, avg 16 min)
- **Labels**: PHQ-8 scores, PTSD indicators
- **License**: Academic/Non-profit research only

### Description
DAIC-WOZ contains clinical interviews designed to support the diagnosis of psychological distress conditions such as anxiety, depression, and PTSD. Data were collected via Wizard-of-Oz interviews with an animated virtual interviewer called "Ellie", controlled by a human interviewer in another room.

### Download Instructions

**Official Access** (Recommended):
1. Visit https://dcapswoz.ict.usc.edu/
2. Complete and sign the data use agreement form
3. Submit the form and wait for approval
4. Download via the provided link after approval

**Requirements**:
- Must be affiliated with an academic institution or non-profit research organization
- Data use agreement must be signed
- Approval process may take several days

### Data Structure
```
DAIC-WOZ/
├── Participant_XXX/
│   ├── audio.wav
│   ├── video.mp4
│   ├── transcript.txt
│   ├── facial_features.csv
│   └── questionnaire_responses.csv
```

### Sample Data Features
- **Audio**: 16kHz WAV files
- **Video**: MP4 format with facial landmarks
- **Text**: Interview transcripts with timestamps
- **Labels**: PHQ-8 depression scores (0-24), binary PTSD indicators

### Relevant Papers
- Used in survey paper: papers/2407.16804_multimodal_ml_mental_health_survey.pdf
- 189 sessions used in AVEC challenges

### Notes
- Consent constraints limit distribution to academics/non-profit researchers only
- Participants are primarily English speakers
- Ages range from late adolescence to older adulthood
- 107 females, 82 males

---

## Dataset 2: E-DAIC (Extended DAIC-WOZ)

### Overview
- **Source**: USC Institute for Creative Technologies
- **Size**: 275 participants, ~70 hours of interviews
- **Format**: Audio, video, transcripts
- **Modalities**: Audio, Video, Text
- **Task**: Depression, PTSD, and anxiety assessment
- **Labels**: PHQ-8, PCL-C (PTSD Checklist)
- **License**: Academic/Non-profit research only

### Description
Extended version of DAIC-WOZ with larger cohort and additional clinical annotations. Includes both human-controlled and autonomous AI interviews.

### Download Instructions

**Official Access**:
1. Visit https://dcapswoz.ict.usc.edu/
2. Select "Extended DAIC (E-DAIC)" option
3. Complete data use agreement
4. Follow same process as DAIC-WOZ

### Sample Features
- 20-minute semi-clinical interviews
- PHQ-8 scores for depression severity
- PCL-C scores for PTSD assessment
- Diverse demographic representation

### Relevant Papers
- Referenced in: papers/2407.16804_multimodal_ml_mental_health_survey.pdf
- Used for AVEC 2019 challenge

### Notes
- Larger sample size than DAIC-WOZ (275 vs 189)
- Includes autonomous AI interview condition
- Limitation: Relatively small number of samples with high PHQ-8 scores

---

## Dataset 3: WESAD (Wearable Stress and Affect Detection)

### Overview
- **Source**: UCI Machine Learning Repository
- **Official Website**: http://archive.ics.uci.edu/ml/datasets/WESAD+(Wearable+Stress+and+Affect+Detection)
- **Size**: 15 participants, ~700MB
- **Format**: Physiological sensor data (PKL format)
- **Modalities**: Physiological signals (ECG, EDA, EMG, Resp, Temp, ACC)
- **Task**: Stress and affect classification
- **Splits**: 15 subjects, lab-based protocol
- **Labels**: Baseline, Stress, Amusement, Meditation
- **License**: Academic/Non-commercial use

### Description
Publicly available dataset for wearable stress and affect detection featuring physiological and motion data from both wrist-worn and chest-worn devices. Includes blood volume pulse, electrocardiogram, electrodermal activity, electromyogram, respiration, body temperature, and 3-axis acceleration.

### Download Instructions

**Method 1: UCI Repository** (Recommended):
```bash
# Direct download
wget https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download -O datasets/WESAD.zip
unzip datasets/WESAD.zip -d datasets/WESAD/
```

**Method 2: Python UCI ML Repo**:
```python
from ucimlrepo import fetch_ucirepo
wesad = fetch_ucirepo(id=465)
# Save to disk
import pickle
with open('datasets/WESAD/wesad_data.pkl', 'wb') as f:
    pickle.dump(wesad, f)
```

**Method 3: Kaggle**:
```bash
# Requires Kaggle API setup
kaggle datasets download -d orvile/wesad-wearable-stress-affect-detection-dataset
unzip wesad-wearable-stress-affect-detection-dataset.zip -d datasets/WESAD/
```

### Data Structure
```
WESAD/
├── S1/
│   ├── S1.pkl  # Contains all sensor data
│   └── S1_quest.csv  # Questionnaire responses
├── S2/
...
├── S15/
└── README.pdf
```

### Sample Data Features
Sensor modalities (chest-worn RespiBAN):
- **ECG**: Electrocardiogram
- **EDA**: Electrodermal activity
- **EMG**: Electromyogram
- **Resp**: Respiration
- **Temp**: Skin temperature
- **ACC**: 3-axis acceleration

Sensor modalities (wrist-worn Empatica E4):
- **BVP**: Blood volume pulse
- **EDA**: Electrodermal activity
- **Temp**: Skin temperature
- **ACC**: 3-axis acceleration

### Loading the Dataset

```python
import pickle
import pandas as pd

# Load subject data
with open('datasets/WESAD/S2/S2.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Access signals
chest_acc = data['signal']['chest']['ACC']  # Chest accelerometer
wrist_eda = data['signal']['wrist']['EDA']  # Wrist EDA
labels = data['label']  # Activity labels

# Labels: 0=not defined, 1=baseline, 2=stress, 3=amusement,
#         4=meditation, 5-7=transition periods
```

### Relevant Papers
- Schmidt et al. (2018): "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection"
- Referenced in survey paper

### Notes
- 15 participants (12 male, 3 female)
- Lab-based controlled protocol
- Sampling rates vary by sensor (700Hz for ECG, 4Hz for EDA, etc.)
- Academic and non-commercial use only

---

## Dataset 4: Mental Health Counseling Conversations (HuggingFace)

### Overview
- **Source**: HuggingFace Datasets
- **Dataset ID**: Amod/mental_health_counseling_conversations
- **Size**: ~100K+ conversation pairs
- **Format**: JSON/CSV
- **Modalities**: Text only
- **Task**: Mental health counseling Q&A
- **License**: Check HuggingFace page for specific license

### Description
A compilation of high-quality, real one-on-one mental health counseling conversations between individuals and licensed professionals, structured as question-answer pairs for fine-tuning language models.

### Download Instructions

**Using HuggingFace datasets library** (Recommended):
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations")

# Save to disk
dataset.save_to_disk("datasets/mental_health_counseling")
```

**Alternative (direct download)**:
```bash
# Clone the dataset repository
git clone https://huggingface.co/datasets/Amod/mental_health_counseling_conversations datasets/mental_health_counseling
```

### Loading the Dataset

```python
from datasets import load_from_disk

# Load from disk
dataset = load_from_disk("datasets/mental_health_counseling")

# Access examples
for example in dataset['train'][:5]:
    print(f"Context: {example['Context']}")
    print(f"Response: {example['Response']}")
    print("---")
```

### Sample Data Structure
```json
[
  {
    "questionID": "1",
    "questionTitle": "Anxiety about relationships",
    "Context": "I've been feeling anxious about...",
    "Response": "It's understandable to feel..."
  }
]
```

### Notes
- Over 100K downloads as of November 2025
- Real counseling conversations (not synthetic)
- Primarily text-based, useful for language modeling
- Can be combined with other modalities for multimodal research

---

## Dataset 5: Additional Datasets Mentioned in Literature

### AVEC 2013/2019 Depression Dataset
- **Access**: Challenge-based, contact organizers
- **Modalities**: Audio, Video
- **Size**: 340 videos (AVEC 2013)

### D-Vlog (Depression Detection from Vlogs)
- **Source**: YouTube clips
- **Modalities**: Audio, Video
- **Access**: May require special request

### SWELL (Stress and Well-being)
- **Size**: 25 participants
- **Modalities**: Physiological sensors
- **Focus**: Knowledge work stress

---

## Comparison Table

| Dataset | Size | Modalities | Task | Age Group | Access |
|---------|------|------------|------|-----------|--------|
| DAIC-WOZ | 189 | A+V+T | Depression/PTSD | Adolescent-Adult | Form required |
| E-DAIC | 275 | A+V+T | Depression/PTSD/Anxiety | Adolescent-Adult | Form required |
| WESAD | 15 | Physiological | Stress/Affect | Adult | Public (UCI) |
| MH Counseling | 100K+ | Text | Counseling Q&A | General | Public (HF) |

**Legend**: A=Audio, V=Video, T=Text, HF=HuggingFace

---

## Ethical Considerations

- All datasets contain sensitive mental health data
- Comply with data use agreements and IRB requirements
- Ensure participant privacy and anonymity
- Use data only for approved research purposes
- Do not redistribute without permission
- Follow GDPR, HIPAA, and other relevant regulations

---

## Citation Information

If you use these datasets, please cite the original papers:

**DAIC-WOZ**:
```
@inproceedings{gratch2014distress,
  title={The distress analysis interview corpus of human and computer interviews},
  author={Gratch, Jonathan and Artstein, Ron and others},
  booktitle={LREC},
  year={2014}
}
```

**WESAD**:
```
@inproceedings{schmidt2018introducing,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and others},
  booktitle={ICMI},
  year={2018}
}
```

---

## Questions?

For questions about dataset access or usage, please refer to the official dataset websites or contact the dataset creators directly.
