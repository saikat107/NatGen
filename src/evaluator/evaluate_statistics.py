import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, wilcoxon
from tqdm import tqdm

from src.evaluator.CodeBLEU.calc_code_bleu import evaluate_per_example


def do_test(target, exp_1, exp_2, test_type, metrics, lang):
    semcode_results, codet5_results = [], []
    with open(target) as goldf:
        with open(exp_1) as semcodef:
            with open(exp_2) as codet5f:
                golds = goldf.readlines()
                for gold, semcode, codet5 in tqdm(
                        zip(golds, semcodef.readlines(), codet5f.readlines()), total=len(golds)
                ):
                    semcode_results.append(
                        evaluate_per_example(
                            reference=gold.strip(),
                            hypothesis=semcode.strip(),
                            lang=lang,
                        )
                    )
                    codet5_results.append(
                        evaluate_per_example(
                            reference=gold.strip(),
                            hypothesis=codet5.strip(),
                            lang=lang,
                        )
                    )
    semcode_df = pd.DataFrame(semcode_results)
    codet5_df = pd.DataFrame(codet5_results)
    if 'all' in metrics:
        metrics = semcode_df.columns.tolist()
    test = ttest_ind if test_type == 't-test' else wilcoxon
    for m in metrics:
        stat, p_value = test(
            semcode_df[m].tolist(), codet5_df[m].tolist(),
            # alternative='greater'
        )
        semcode_r = semcode_df[m].tolist()
        codet5_r = codet5_df[m].tolist()
        result_data = pd.DataFrame(
            [
                {'Experiment': 'SemCode', m: v} for v in semcode_df[m].tolist()
            ] + [
                {'Experiment': 'CodeT5', m: v} for v in codet5_df[m].tolist()
            ]
        )
        print(
            """SemCode : 
            Mean    %.4f
            1st q   %.4f 
            Median  %.4f
            3rd q   %.4f""" % (
                np.mean(semcode_r).item(),
                np.percentile(semcode_r, 25).item(),
                np.percentile(semcode_r, 50).item(),
                np.percentile(semcode_r, 75).item(),
            )
        )
        print(
            """CodeT5 : 
            Mean    %.4f
            1st q   %.4f 
            Median  %.4f
            3rd q   %.4f""" % (
                np.mean(codet5_r).item(),
                np.percentile(codet5_r, 25).item(),
                np.percentile(codet5_r, 50).item(),
                np.percentile(codet5_r, 75).item(),
            )
        )
        # print(result_data.columns)
        plt.figure()
        ax = sns.violinplot(
            data=result_data,
            x=m, y='Experiment', hue='Experiment',
            split=True,
            inner='quartiles'
        )
        ax.set(title=m)
        plt.legend()
        r_data = pd.DataFrame(
            {
                'SemCode': semcode_df[m].tolist(),
                'CodeT5': codet5_df[m].tolist()
            }
        )
        plt.figure()
        sns.jointplot(data=r_data, x='SemCode', y='CodeT5')
        plt.show()
        print(
            f"""
            Metric     : {m} 
            Statistics : {stat}
            P-Value    : {p_value}"""
        )
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", help="Path of the target file", required=True)
    parser.add_argument(
        "--exp_1", help="Path of the experiment 1 output file. Usually the Semcode output", required=True
    )
    parser.add_argument(
        "--exp_2", help="Path of the experiment 2 output file. Usually the CodeT5 output", required=True
    )
    parser.add_argument("--test_type", choices=['t-test', 'wilcoxon'], default='t-test')
    parser.add_argument("--lang", required=True)
    parser.add_argument("--metrics", nargs='+', default=['all'])
    args = parser.parse_args()
    do_test(
        target=args.gold,
        exp_1=args.exp_1,
        exp_2=args.exp_2,
        test_type=args.test_type,
        metrics=args.metrics,
        lang=args.lang
    )
