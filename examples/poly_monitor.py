import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse

from shrinkbench.plot import df_from_results, plot_df, save_to_pdf
sns.set_theme(style="darkgrid")

parser = argparse.ArgumentParser(description="Monitor scripts for shrinkbench results")
parser.add_argument('-g', '--glob', default="*")
args = parser.parse_args()


df, logs, params = df_from_results('results', glob=args.glob)

fig, ax = plt.subplots(figsize=(7, 5))

for i, (exp_log, exp_param) in enumerate(zip(logs, params)):
    print("")
    exp_result = df.iloc[i]
    print('{}, sparsity = {}, acc1 = {}'.format(
        exp_result["strategy"],
        1-1/exp_result["real_compression"],
        exp_result["post_acc1"]
    ))
    args = exp_param["pruning_kwargs"]
    print(exp_param["train_kwargs"])
    print(args["scheduler"], args["scheduler_args"])

    label = "-".join([ str(i) for i in list(args['scheduler_args'].values())[1:] ])
    sns.lineplot(ax=ax, data=exp_log, x='epoch', y='val_acc1', label=label, marker="o")

save_to_pdf(name="train.pdf")
