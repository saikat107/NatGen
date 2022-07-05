import argparse
import json
import multiprocessing
import os
import pickle
import random
from multiprocessing import Pool, cpu_count

import nltk
import numpy as np
import torch
from tqdm import tqdm

from src.data_preprocessors.transformations import (
    NoTransformation, SemanticPreservingTransformation,
    BlockSwap, ConfusionRemover, DeadCodeInserter,
    ForWhileTransformer, OperandSwap, VarRenamer
)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_transformers_from_conf_file(processing_conf):
    classes = [BlockSwap, ConfusionRemover, DeadCodeInserter, ForWhileTransformer, OperandSwap, VarRenamer]
    transformers = {
        c: processing_conf[c.__name__] for c in classes
    }
    return transformers


class ExampleProcessor:
    def __init__(
            self,
            language,
            parser_path,
            transformation_config,
            bidirection_transformation=False,
            max_function_length=400
    ):
        self.language = language
        self.parser_path = parser_path
        self.transformation_config = transformation_config
        self.max_function_length = max_function_length
        self.bidirection_transformation = bidirection_transformation

    def initialize(self):
        global example_tokenizer
        global example_transformer
        transformers = create_transformers_from_conf_file(self.transformation_config)
        if self.language == "nl":
            example_tokenizer = nltk.word_tokenize
        else:
            example_tokenizer = NoTransformation(self.parser_path, self.language)
        example_transformer = SemanticPreservingTransformation(
            parser_path=self.parser_path, language=self.language, transform_functions=transformers
        )

    def process_example(self, code):
        global example_tokenizer
        global example_transformer
        try:
            if self.language == "nl":
                original_code = " ".join(example_tokenizer(code))
            else:
                original_code, _ = example_tokenizer.transform_code(code)
            if len(original_code.split()) > self.max_function_length:
                return -1
            transformed_code, used_transformer = example_transformer.transform_code(code)
            if used_transformer:
                if used_transformer == "ConfusionRemover":  # Change the direction in case of the ConfusionRemover
                    temp = original_code
                    original_code = transformed_code
                    transformed_code = temp
                if isinstance(self.bidirection_transformation, str) and self.bidirection_transformation == 'adaptive':
                    bidirection = (used_transformer in ["BlockSwap", "ForWhileTransformer", "OperandSwap"])
                else:
                    assert isinstance(self.bidirection_transformation, bool)
                    bidirection = self.bidirection_transformation
                if bidirection and np.random.uniform() < 0.5 \
                        and used_transformer != "SyntacticNoisingTransformation":
                    return {
                        'source': original_code,
                        'target': transformed_code,
                        'transformer': used_transformer
                    }
                else:
                    return {
                        'source': transformed_code,
                        'target': original_code,
                        'transformer': used_transformer
                    }
            else:
                return -1
        except KeyboardInterrupt:
            print("Stopping parsing for ", code)
            return -1
        except:
            return -1


def process_functions(
        pool, example_processor, functions,
        train_file_path=None, valid_file_path=None, valid_percentage=0.002
):
    used_transformers = {}
    success = 0
    tf = open(train_file_path, "wt") if train_file_path is not None else None
    vf = open(valid_file_path, "wt") if train_file_path is not None else None
    with tqdm(total=len(functions)) as pbar:
        processed_example_iterator = pool.imap(
            func=example_processor.process_example,
            iterable=functions,
            chunksize=1000,
        )
        count = 0
        while True:
            pbar.update()
            count += 1
            try:
                out = next(processed_example_iterator)
                if isinstance(out, int) and out == -1:
                    continue
                if out["transformer"] not in used_transformers.keys():
                    used_transformers[out["transformer"]] = 0
                used_transformers[out["transformer"]] += 1
                if np.random.uniform() < valid_percentage:
                    if vf is not None:
                        vf.write(json.dumps(out) + "\n")
                        vf.flush()
                else:
                    if tf is not None:
                        tf.write(json.dumps(out) + "\n")
                        tf.flush()
                success += 1
            except multiprocessing.TimeoutError:
                print(f"{count} encountered timeout")
            except StopIteration:
                print(f"{count} stop iteration")
                break
    if tf is not None:
        tf.close()
    if vf is not None:
        vf.close()
    print(
        f"""
            Total   : {len(functions)}, 
            Success : {success},
            Failure : {len(functions) - success}
            Stats   : {json.dumps(used_transformers, indent=4)}
            """
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--langs', default=["java"], nargs='+',
        help="Languages to be processed"
    )
    parser.add_argument(
        '--input_dir', default="/home/saikatc/HDD_4TB/from_server/SemCode/pretraining/data/raw", help="Directory of the language pickle files"
    )
    parser.add_argument(
        '--output_dir', default="/home/saikatc/HDD_4TB/from_server/SemCode/pretraining/data/processed", help="Directory for saving processed code"
    )
    parser.add_argument(
        '--processing_config_file', default="configs/data_processing_config.json",
        help="Configuration file for data processing."
    )
    parser.add_argument(
        '--parser_path', help="Tree-Sitter Parser Path", default="/home/saikatc/HDD_4TB/from_server/SemCode/parser/languages.so"
    )
    parser.add_argument(
        "--workers", help="Number of worker CPU", type=int, default=20
    )
    parser.add_argument(
        "--timeout", type=int, help="Maximum number of seconds for a function to process.", default=10
    )
    parser.add_argument(
        "--valid_percentage", type=float, help="Percentage of validation examples", default=0.001
    )
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()
    set_seeds(args.seed)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    configuration = json.load(open(args.processing_config_file))
    print(configuration)
    for lang in args.langs:
        print(f"Now processing : {lang}")
        pkl_file = os.path.join(args.input_dir, lang + ".pkl")
        data = pickle.load(open(pkl_file, "rb"))
        functions = [ex['function'] for ex in data]
        # for f in functions[:5]:
        #     print(f)
        #     print("=" * 100)
        if lang == "php":
            functions = ["<?php\n" + f + "\n?>" for f in functions]
        example_processor = ExampleProcessor(
            language=lang,
            parser_path=args.parser_path,
            transformation_config=configuration["transformers"],
            max_function_length=(
                configuration["max_function_length"] if "max_function_length" in configuration else 400
            ),
            bidirection_transformation=(
                configuration["bidirection_transformation"] if "bidirection_transformation" in configuration else False
            )
        )
        pool = Pool(
            processes=min(cpu_count(), args.workers),
            initializer=example_processor.initialize
        )
        process_functions(
            pool=pool,
            example_processor=example_processor,
            functions=functions,
            train_file_path=os.path.join(out_dir, f"{lang}_train.jsonl"),
            valid_file_path=os.path.join(out_dir, f"{lang}_valid.jsonl"),
            valid_percentage=args.valid_percentage
        )
        del pool
        del example_processor
