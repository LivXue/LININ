# LININ: Logic Integrated Neural Inference Network for Explanatory Visual Question Answering

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-10.1109-blue.svg?logo=ieee)](https://ieeexplore.ieee.org/document/10814657)
[![Python](https://img.shields.io/badge/Python-3.7+-yellow.svg?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📄 Paper
This repository contains the official implementation of the paper:  
**[LININ: Logic Integrated Neural Inference Network for Explanatory Visual Question Answering](https://ieeexplore.ieee.org/document/10814657)**  
Published at IEEE Transactions on Multimedia, 2024

---

## 🌟 Introduction
This is the authors' official implementation of **LININ (Logic Integrated Neural Inference Network)**, a neural inference network that integrates logic reasoning for explanatory visual question answering.

---

## 📊 Data Preparation

### Required Datasets
1. **GQA Dataset**: Download from [GQA official website](https://cs.stanford.edu/people/dorarad/gqa/download.html)
2. **GQA-OOD Dataset**: Download from [GQA-OOD GitHub](https://github.com/gqa-ood/GQA-OOD)
3. **Bottom-up Features**: Download from [LXMERT GitHub](https://github.com/airsplay/lxmert) and unzip it

### Data Processing Steps

#### Step 1: Extract Features
> **Important**: This code needs to be run on Linux
```bash
python ./preprocessing/extract_tsv.py --input $TSV_FILE --output $FEATURE_DIR
```

#### Step 2: GQA-REX Annotations
We provide the GQA-REX Dataset annotations in:
- `model/processed_data/converted_explanation_train_balanced.json`
- `model/processed_data/converted_explanation_val_balanced.json`

(Optional) You can also construct the GQA-REX Dataset by following the [original instructions](https://github.com/szzexpoi/rex).

#### Step 3: Clean Data
```bash
python ./preprocessing/clean_questions.py
```

#### Step 4: Generate Answer Masks
Run our FOL-based question analysis program:
```bash
python ./preprocessing/generate_ans_mask.py
```

---

## 🧠 Models

We provide four model implementations in `model/model/model.py`.

---

## 🚀 Training &amp; Evaluation

### Preprocessing: Generate Dictionary
Before training, generate dictionaries for questions, answers, and explanations:
```bash
cd ./model
python generate_dictionary --question $GQA_ROOT/question --exp $EXP_DIR --save ./processed_data
```

### Training
```bash
python main.py --mode train \
    --anno_dir $GQA_ROOT/question \
    --ood_dir $OOD_ROOT/data \
    --sg_dir $GQA_ROOT/scene_graph \
    --lang_dir ./processed_data \
    --img_dir $FEATURE_DIR/features \
    --bbox_dir $FEATURE_DIR/box \
    --checkpoint_dir $CHECKPOINT \
    --explainable True
```

### Evaluation
To evaluate on GQA-testdev or generate submission files for test-standard:
```bash
python main.py --mode $MODE \
    --anno_dir $GQA_ROOT/question \
    --ood_dir $OOD_ROOT/data \
    --lang_dir ./processed_data \
    --img_dir $FEATURE_DIR/features \
    --weights $CHECKPOINT/model_best.pth \
    --explainable True
```

Set `$MODE` to `eval` or `submission` accordingly.

---

## 📖 Citation
If you find this work useful, please cite our paper:

```bibtex
@article{xue2024linin,
  title={LININ: Logic Integrated Neural Inference Network for Explanatory Visual Question Answering},
  author={Xue, Dizhan and Qian, Shengsheng and Fang, Quan and Xu, Changsheng},
  journal={IEEE Transactions on Multimedia}, 
  year={2024},
  volume={27},
  pages={16-27},
  doi={10.1109/TMM.2024.3521709}
}
```

---

## 📝 License
This project is licensed under the MIT License.

---

**Note**: Please fill in the citation information with your paper's details.
