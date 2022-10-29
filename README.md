# Out-of-Distribution Generalization Challenge in Dialog State Tracking
This repository contains official implementation of the paper *(Out-of-Distribution Generalization Challenge in Dialog State Tracking)[https://openreview.net/forum?id=Z-k91NB8Eh]*.

## Abstract

Dialog State Tracking (DST) is a core component for multi-turn Task-Oriented Dialog (TOD) systems to understand the dialogs. DST models need to generalize to Out-of-Distribution (OOD) utterances due to the open environments dialog systems face. Unfortunately, utterances in TOD are multi-labeled, and most of them appear in specific contexts (i.e., the dialog histories). Both characteristics make them different from the conventional focus of OOD generalization research and remain unexplored. In this paper, we formally define OOD utterances in TOD and evaluate the generalizability of existing competitive DST models on the OOD utterances. Our experimental result shows that the performance of all models drops considerably in dialogs with OOD utterances, indicating an OOD generalization challenge in DST.

## Installation

This code is written with python 3.8.3. For packages required to run this code, please refer to `requirements.txt`.

## Preparation

Please download the MultiWOZ 2.3 data set from [their official repository](https://github.com/lexmen318/MultiWOZ-coref) and place the downloaded under the `datasets` folder.

As expected, you should have the following directory structure.
```
DST_OOD
├── datasets
│   ├── MultiWOZ2_3
│   │   ├── data.json
│   │   ├── dialogue_acts.json
│   │   └── ontology.json
│   └── ... 
└── ...
```
Then, prepare the training data and OOD test data by running
```bash
cd datasets
python DataInit.py
```
This will create `MultiWOZ_OoD` under the `datasets` folder. Besides training and validation data copied from the original datasets, `MultiWOZ_OoD` contains test data of different types both from the original test set and by generation (i.e., the MultiWOZ OOD test set described in our paper). 

The script also preprocesses the data for three different DST methods (i.e., SimpleTOD, Trippy and TRADE) and saves the preprocessed data in `MultiWOZ_OoD_${method}`.

## Training

To train the models, run
```bash
python experiments.py --method SimpleTOD --devices 0 --pretrained gpt2 --train

python experiments.py --method Trippy --devices 0 --pretrained bert-base-uncased --train

python experiments.py --method Trade --devices 0 --pretrained none --train
```
You can accelerate training with data distributed parallel training by assigning multiple devices. For example, 
```bash
python experiments.py --method SimpleTOD --devices 0 1 2 3 --pretrained gpt2 --train
```

By default, the model checkpoints are save under `${method}/checkpoints/`.

We also shared our trained checkpoints at xxx(TBD).

## Testing

To test the models, run
```bash
python experiments.py --method SimpleTOD --devices 0 --pretrained gpt2 --ood --checkpoint SimpleTOD/checkpoint/${ckpt_folder}

python experiments.py --method Trippy --devices 0 --pretrained bert-base-uncased --ood --checkpoint Trippy/checkpoint/${ckpt_folder}

python experiments.py --method Trade --devices 0 --pretrained none --ood --checkpoint Trade/checkpoint/${ckpt_folder}
```

## License
The code is released under BSD 3-Clause - see [LICENSE](https://github.com/yegcjs/DST_OOD/blob/main/license) for details.

This code includes other open source codes from [SimpleTOD](https://github.com/salesforce/simpletod), [Trippy](gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) and [Trade](https://github.com/jasonwu0731/trade-dst). These components have their own liscences. Please refer to their official repositories.
