# Wav2ToBI

***Under Construction***

This repository contains code used in the paper "[Wav2ToBI: a new approach to automatic ToBI transcription][paper-link]" accepted at Interspeech 2023.

## Setup

1. Install Pytorch following the instructions stated on [Official Pytorch Website][torch-link].

2. Create conda environment with python version 3.8

```
conda create --name wav2tobi python=3.8
```

3. Install the required packages in `requirements.txt`
```
pip install -r requirements.txt
```

4. (Optional) Dataset
The datasets used in the paper were the Boston University Radio News Corpus (BURNC) and the Boston Directions Corpus (BDC). Datasets should be stored under `data/` with different types of required files in their respective folder. The files required for intonational phrase boundary detection should be the .wav files and the .ton files, whereas the files required for pitch accent detection should be the .wav files and the .brk files. 

## Intonational Phrase Boundary Detection


### Training Pipeline

1. Data Preprocessing

The break annotation files should be provided under `data/break_files`, and the wav files should be provided
under `data/wav_files`. The names of break files should match the names of the wav files as shown in the sample.

```
python src/break_preprocess.py -h

usage: break_preprocess.py [-h] [--bfilepath BFILEPATH] [--wfilepath WFILEPATH] 
                           [--sec_per_split SEC_PER_SPLIT] [--window_size WINDOW_SIZE]
                           [--output_path OUTPUT_PATH] (--peak | --flat)
```
**Currently Only Supports Fuzzy Labeling**

2. Model Training

The trained checkpoints are saved in Huggingface Hub. 

```
python src/train.py -h

usage: train.py [-h] [--model_checkpoint MODEL_CHECKPOINT] [--file_train FILE_TRAIN] 
                [--file_valid FILE_VALID] [--file_eval FILE_EVAL] [--num_epochs NUM_EPOCHS] 
                [--batch_size BATCH_SIZE] [--file_output FILE_OUTPUT]
                [--model_save_dir MODEL_SAVE_DIR] [--max_duration MAX_DURATION] [--mode MODE]
                [--epochs_between_checkpoints EPOCHS_BETWEEN_CHECKPOINTS] [--lr_init LR_INIT]
                [--lr_num_warmup_steps LR_NUM_WARMUP_STEPS] 
                [--remove_last_label REMOVE_LAST_LABEL]

```

When running with tag `--mode eval` or `--mode both`, the model will output the file containing the results 
when tested on the test data on the model, and will be saved in `FILE_OUTPUT`.

3. Model Evaluation

```
python src/eval.py -h

usage: eval.py [-h] [--file_eval FILE_EVAL] [--file_test FILE_TEST] [--file_ind FILE_IND] 
               [--file_out FILE_OUT] (--peak | --flat)
```
Model evaluation requires the file containing ground truth `FILE_EVAL` (which can be obtained through 
`src/preprocess.py`), and the file containing prediction results from the ground truth `FILE_TEST` (which 
can be obtained from `src/train.py`). An example figure on `FILE_IND` will be outputed, containing the 
comparison results between model prediction and the ground truth.


## Pitch Accent Detection

### Training Pipeline

1. Data Preprocessing

The tone annotation files should be provided under `data/tone_files`, and the wav files should be provided
under `data/wav_files`. The names of tone files should match the names of the wav files as shown in the sample.

```
python src/tone_preprocess.py -h

usage: tone_preprocess.py [-h] [--tfilepath TFILEPATH] [--wfilepath WFILEPATH] 
                          [--sec_per_split SEC_PER_SPLIT] [--window_size WINDOW_SIZE]
                          [--output_path OUTPUT_PATH] [--splitwav] (--peak | --flat)
```

2. Model Training

The trained checkpoints are saved in Huggingface Hub. 

```
python src/train.py -h

usage: train.py [-h] [--model_checkpoint MODEL_CHECKPOINT] [--file_train FILE_TRAIN] 
                [--file_valid FILE_VALID] [--file_eval FILE_EVAL] [--num_epochs NUM_EPOCHS] 
                [--batch_size BATCH_SIZE] [--file_output FILE_OUTPUT]
                [--model_save_dir MODEL_SAVE_DIR] [--max_duration MAX_DURATION] [--mode MODE]
                [--epochs_between_checkpoints EPOCHS_BETWEEN_CHECKPOINTS] [--lr_init LR_INIT]
                [--lr_num_warmup_steps LR_NUM_WARMUP_STEPS] 
                [--remove_last_label REMOVE_LAST_LABEL]

```

When running with tag `--mode eval` or `--mode both`, the model will output the file containing the results 
when tested on the test data on the model, and will be saved in `FILE_OUTPUT`.

3. Model Evaluation

```
python src/eval.py -h

usage: eval.py [-h] [--file_eval FILE_EVAL] [--file_test FILE_TEST] [--file_ind FILE_IND] 
               [--file_out FILE_OUT] (--peak | --flat)
```
Model evaluation requires the file containing ground truth `FILE_EVAL` (which can be obtained through 
`src/preprocess.py`), and the file containing prediction results from the ground truth `FILE_TEST` (which 
can be obtained from `src/train.py`). An example figure on `FILE_IND` will be outputed, containing the 
comparison results between model prediction and the ground truth.

## Prosody Boundary Prediction using Existing Checkpoint

To perform prosody boundary prediction using existing checkpoints, follow the same process
for data preprocessing as stated in step 1 for both processes. 

#### Prosody Prediction

Checkpoints reported were uploaded into huggingface hub. Find the desired model among
`ReginaZ/Wav2ToBI-PA-Flat`, `ReginaZ/Wav2ToBI-PA-Fuzzy`, `ReginaZ/Wav2ToBI-PB-Fuzzy`,
`ReginaZ/Wav2ToBI-PB-Flat`.

*Note*: The checkpoint for pitch accent detection has lstm hidden size of 128, whereas
the checkpoint for phrase boundary prediction has hidden size of 256. The parameter can 
be changed in `src/model.py`

```
python src/predict.py -h
usage: predict.py [-h] [--model_checkpoint MODEL_CHECKPOINT] [--input_path INPUT_PATH] 
                  [--file_eval FILE_EVAL] [--file_ind FILE_IND]
                  [--file_out FILE_OUT] [--batch_size BATCH_SIZE] [--max_duration MAX_DURATION] 
                  (--peak | --flat)

```

## Acknowledgement
The code is greatly inspired by [mkunes/w2v2_audioFrameClassification][github-link] used in Kunešová, M., Řezáčková, M. (2022). Detection of Prosodic Boundaries in Speech Using Wav2Vec 2.0. In: International Conference on Text, Speech, and Dialogue (TSD 2022). LNAI vol. 13502, pp. 377-388. Springer. doi: 10.1007/978-3-031-16270-1_31 

[paper-link]: https://www.interspeech2023.org

[github-link]: https://github.com/mkunes/w2v2_audioFrameClassification.git

[torch-link]: https://pytorch.org