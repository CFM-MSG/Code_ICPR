Reproducible Code for our work **Intrinsic Consistency Preservation with Adaptively Reliable Sample for Source-free Domain Adaptation**.

## Installation and requirements
```
python == 3.7
pytorch == 1.10
cudatoolkit == 11.1
torchvision == 0.11
```

## Data preparation
Office, Office-Home, VisDA-C, Office-Home (RSUT), VisDA-C (RSUT), and DomainNet can be found in [ISFDA](https://github.com/LeoXinhaoLee/Imbalanced-Source-free-Domain-Adaptation).
Note that these datasets are publicly available and have been widely used in the community.

## Training and adaptation
Taking the VisDA dataset as an example, the training command is:
```
bash image_source.sh
```
The adaptation command is:
```
bash image_target.sh
```

## Acknowledge
Our code partially follows the programming style of [NRC](https://github.com/Albert0147/SFDA_neighbors) and [SHOT](https://github.com/tim-learn/SHOT).
Thanks for the excellent work of NRC and SHOT.
