# LININ: Logic Integrated Neural Inference Network for Explanatory Visual Question Answering

## Introduction
This is the authors' implementation of LININ (Logic Integrated Neural Inference Network).

### Data
1. Download the [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).
2. Download the [GQA-OOD Dataset](https://github.com/gqa-ood/GQA-OOD)
3. Download the [bottom-up features](https://github.com/airsplay/lxmert) and unzip it.
4. Extracting features from the raw tsv files (**Important**: You need to run the code in Linux):
  ```
  python ./preprocessing/extract_tsv.py --input $TSV_FILE --output $FEATURE_DIR
  ```
5. We provide the annotations of GQA-REX Dataset in `model/processed_data/converted_explanation_train_balanced.json` and `model/processed_data/converted_explanation_val_balanced.json`.
6. (Optional) You can construct the GQA-REX Dataset by yourself following [instructions by its authors](https://github.com/szzexpoi/rex).
7. Clean data using our script:
  ```
  python ./preprocessing/clean_questions.py
  ```
8. Run our FOL-based question analysis program to generate answer masks:
  ```
  python ./preprocessing/generate_ans_mask.py
  ```

### Models
We provide four models in `model/model/model.py`.

### Training and Test
Before training, you need to first generate the dictionary for questions, answers, and explanations:
  ```
  cd ./model
  python generate_dictionary --question $GQA_ROOT/question --exp $EXP_DIR --save ./processed_data
  ```

The training process can be called as:
  ```
  python main.py --mode train --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --sg_dir $GQA_ROOT/scene_graph --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --bbox_dir $FEATURE_DIR/box --checkpoint_dir $CHECKPOINT --explainable True
  ```
To evaluate on the GQA-testdev set or generating submission file for online evaluation on the test-standard set, call:
  ```
  python main.py --mode $MODE --anno_dir $GQA_ROOT/question --ood_dir $OOD_ROOT/data --lang_dir ./processed_data --img_dir $FEATURE_DIR/features --weights $CHECKPOINT/model_best.pth --explainable True
  ```
and set `$MODE` to `eval` or `submission` accordingly.