#!/bin/bash

function setup_repo() {
    mkdir -p sitter-libs;
    git clone https://github.com/tree-sitter/tree-sitter-go sitter-libs/go;
    git clone https://github.com/tree-sitter/tree-sitter-javascript sitter-libs/js;
    git clone https://github.com/tree-sitter/tree-sitter-c sitter-libs/c;
    git clone https://github.com/tree-sitter/tree-sitter-cpp sitter-libs/cpp;
    git clone https://github.com/tree-sitter/tree-sitter-c-sharp sitter-libs/cs;
    git clone https://github.com/tree-sitter/tree-sitter-python sitter-libs/py;
    git clone https://github.com/tree-sitter/tree-sitter-java sitter-libs/java;
    git clone https://github.com/tree-sitter/tree-sitter-ruby sitter-libs/ruby;
    git clone https://github.com/tree-sitter/tree-sitter-php sitter-libs/php;
    mkdir -p "parser";
    python3 create_tree_sitter_parser.py sitter-libs;
    cp parser/languages.so src/evaluator/CodeBLEU/parser/languages.so
}

function create_and_activate() {
    conda create --name natgen python=3.6;
    conda activate natgen;
}

function install_deps() {
    #conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
    conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    conda install datasets==1.18.3 -c conda-forge
    conda install transformers==4.16.2 -c conda-forge
    conda install tensorboard==2.8.0 -c conda-forge
    pip install tree-sitter==0.19.0;
    pip install nltk==3.6.7;
    pip install scipy==1.5.4;
    # Please add the command if you add any package.
}

#create_and_activate;
install_deps;
#setup_repo;
