import argparse
import logging
import multiprocessing
import os
import time
import torch
from src.finetuning.configs import add_args, set_seed, set_dist
from src.finetuning.models import build_or_load_gen_model
from src.finetuning.utils import get_elapse_time, load_pretrain_eval_data
from src.finetuning.generation import eval_ppl_epoch, eval_bleu_epoch, eval_bleu_per_example
import pandas as pd
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro_eval", action='store_true')
    args = add_args(parser)
    logger.info(args)
    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    logger.info("  " + "***** Evaluating Pretrain Performance *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    data_per_lang, (complete_examples, complete_data) = load_pretrain_eval_data(
        args=args, pretrain_datadir=args.data_dir, pool=pool, tokenizer=tokenizer
    )
    if not args.macro_eval:
        eval_ppl = eval_ppl_epoch(args, complete_data, complete_examples, model, tokenizer)
        logger.info(f"{eval_ppl}")
        eval_result = eval_bleu_per_example(
            args=args, eval_data=complete_data, eval_examples=complete_examples, model=model, tokenizer=tokenizer,
            split_tag='test', criteria='Pretrain-Evaluation'
        )
        eval_result_df = pd.DataFrame(data=eval_result)
        output_fn = os.path.join(args.res_dir, "pretraining_test_res.csv")
        eval_result_df.to_csv(output_fn)
        logger.info(f"Detailed Output written to : {os.path.realpath(output_fn)}")
    else:
        fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
        for lang in data_per_lang.keys():
            args.lang = lang
            eval_examples, eval_data = data_per_lang[lang]
            eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
            logger.info(f"{lang} : {eval_ppl}")
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', f"{args.task}")
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu']
            result_str = "Lang : %s PPL: %.5f bleu-4: %.2f, " \
                         "em: %.4f, codebleu: %.4f\n" % (
                             lang, eval_ppl, test_bleu, test_em, test_codebleu
                         )
            logger.info(result_str)
            print(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write(result_str)
        fa.close()


if __name__ == "__main__":
    main()
