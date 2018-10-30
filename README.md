# NBTNGMA4ED
This is the source code for the paper ''Collective Event Detection via a Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms'' accepted by EMNLP2018.

## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- numpy
- python 3.5

## Usage
### Download datasets
The ACE 2005 dataset can be download from：https://catalog.ldc.upenn.edu/LDC2006T06
 
### Default parameters:
- batch size: 20
- gradient clip: 5
- embedding size: 100
- optimizer: Adam
- dropout rate: 0.5
- learning rate: 0.001
- seg_dim:     20,
- subtype_dim: 20,
- num_steps:   40,
- lstm_dim1:    100
- lstm_dim2:    100
- seq_size:    8
- alpha:       5

### Document 
- doc_data: train\test\dev data
- doc_dict: document information and its corresponding ID
- 100.utf8: pretrained word embedding
- doc_id:   document id of train\test\dev data 

### Data format
The training and test data is expected in standard tab-separated format. One word per line, separate column for token and label, empty line between sentences.
The first column is assumed to be the token and the last column is the label. The second column is the current document ID.The third column is the type of the entity. The fourth column is the entity subtyp of the entity. For example:

by		CNN_IP_20030405.1600.01-1		O		O		O
coalition		CNN_IP_20030405.1600.01-1		B-1_GPE		B-2_Nation		O
air		CNN_IP_20030405.1600.01-1		O		O		B-Conflict_Attack
strikes		CNN_IP_20030405.1600.01-1		O		O		I-Conflict_Attack



### Train and test the model
- $ train - python3 main.py --train=True
- $ test - python3 main.py --train=False

### Citation
If you use the code, please cite this paper:

Yubo Chen, Hang Yang, Kang Liu, Jun Zhao, Yantao Jia. Collective Event Detection via a Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP2018).


