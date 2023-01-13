# NatGen: Generative Pre-training by "Naturalizing" Source Code.
Saikat Chakraborty, Toufique Ahmed, Yangruibo Ding, Premkumar T Devanbu, Baishakhi Ray. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE ’22), November 14-18, 2022, Singapore, Singapore. ACM, New York, NY, USA, 13 pages. [https://doi.org/10.1145/3540250.3549162](https://doi.org/10.1145/3540250.3549162).

<br/>

<p align="center">
  <a href="https://github.com/saikat107/NatGen/issues-raw">
    <img src="https://img.shields.io/github/issues-raw/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/issues-closed-raw">
    <img src="https://img.shields.io/github/issues-closed-raw/saikat107/NatGen" /> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/issues-pr-raw">
    <img src="https://img.shields.io/github/issues-pr-raw/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/issues-pr-closed-raw">
    <img src="https://img.shields.io/github/issues-pr-closed-raw/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/network/members">
    <img src="https://img.shields.io/github/forks/saikat107/NatGen"/> 
  </a>  
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/stargazers">
    <img src="https://img.shields.io/github/stars/saikat107/NatGen"/> 
  </a>
  &nbsp;
  <a href="https://github.com/saikat107/NatGen/LICENSE">
    <img src="https://img.shields.io/github/license/saikat107/NatGen"/> 
  </a> 
  &nbsp;
  <img src="https://img.shields.io/github/languages/count/saikat107/NatGen"/>
  &nbsp;
  <img src="https://img.shields.io/github/languages/top/saikat107/NatGen"/>
  &nbsp;
  <img src="https://img.shields.io/github/last-commit/saikat107/NatGen"/>
</p>

### <p align="center">[The paper](https://dl.acm.org/doi/abs/10.1145/3540250.3549162) &emsp; [Slide Deck](https://docs.google.com/presentation/d/1T6kjiohAAR1YvcNvTASR94HptA3xHGCl/edit?usp=sharing&ouid=111755026725574085503&rtpof=true&sd=true)</p>

## Getting Started

### Environment Requirements
```
pytorch==1.7.0 
cudatoolkit=11.1
datasets==1.18.3
transformers==4.16.2
tensorboard==2.8.0
tree-sitter==0.19.0;
nltk==3.6.7;
scipy==1.5.4;
```

To setup the environment. Please uncomment [line 35 and 36](setup.sh#L35-36) (or run those code in your shell).
```
bash run setup.sh
```

### Download and preprocess the training data
```
cd scripts/pretraining;
bash process_data.sh
```
Data processing takes several parameters. These parameters are passed through a configuration json file. The configuration file should be in [configs/pretraining/data_config](configs/pretraining/data_config) directory. 

### Pretrain the model 
```
cd scripts/pretraining;
bash train.sh <EXPERIMENT_NAME> <GPUS>
```
Adjust the `per_device_train_batch_size` and `gradient_accumulation_steps` and number of GPUS using to get the final effective batch size in the [training arguments json file](configs/pretraining/train_config/default_train_args.json). 
`per_device_train_batch_size * gradient_accumulation_steps * number of gpus`. We use distributed training to pre-train. 

We reused source code from various open source code repositories
1. [CodeT5](https://github.com/salesforce/CodeT5)
2. [Microsoft CodeXGLUE](https://github.com/microsoft/CodeXGLUE)
Out sincere thanks to the authors of these repositories for open-sourcing their work. 

# Citation
If you use  this repository, please cite,
```
@inproceedings{chakraborty2022natgen,
    author = {Chakraborty, Saikat and Ahmed, Toufique and Ding, Yangruibo and Devanbu, Premkumar T. and Ray, Baishakhi},
    title = {NatGen: Generative Pre-Training by “Naturalizing” Source Code},
    year = {2022},
    isbn = {9781450394130},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3540250.3549162},
    doi = {10.1145/3540250.3549162},
    booktitle = {Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
    pages = {18–30},
    numpages = {13},
    keywords = {Neural Network, Semantic Preserving Transformation, Source Code Transformer, Source Code Pre-training},
    location = {Singapore, Singapore},
    series = {ESEC/FSE 2022}
}
```
