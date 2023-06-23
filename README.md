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

## Intonational Phrase Boundary Detection

## Pitch Accent Detection


## Acknowledgement
The code is greatly inspired by [mkunes/w2v2_audioFrameClassification][github-link] used in Kunešová, M., Řezáčková, M. (2022). Detection of Prosodic Boundaries in Speech Using Wav2Vec 2.0. In: International Conference on Text, Speech, and Dialogue (TSD 2022). LNAI vol. 13502, pp. 377-388. Springer. doi: 10.1007/978-3-031-16270-1_31 

[paper-link]: https://www.interspeech2023.org

[github-link]: https://github.com/mkunes/w2v2_audioFrameClassification.git

[torch-link]: https://pytorch.org