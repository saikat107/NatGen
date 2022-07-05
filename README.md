## NatGen: Generative pre-training by “Naturalizing” source code

### Getting Started

#### Environment Requirements
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

#### Download and preprocess the training data
```
cd scripts/pretraining;
bash process_data.sh
```
Data processing takes several parameters. These parameters are passed through a configuration json file. The configuration file should be in [configs/pretraining/data_config](configs/pretraining/data_config) directory. 

#### Pretrain the model 
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
