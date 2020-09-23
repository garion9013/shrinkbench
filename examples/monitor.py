import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import argparse

from shrinkbench.plot import df_from_results, plot_df, save_to_pdf
sns.set_theme(style="darkgrid")

parser = argparse.ArgumentParser(description="Monitor scripts for shrinkbench results")
parser.add_argument('-g', '--glob', default="*")
parser.add_argument('-p', '--path', default="")
args = parser.parse_args()


df, logs, params = df_from_results(f'{args.path}', glob=args.glob)

fig, ax = plt.subplots(len(logs), figsize=(5, 10))

# If there exists only a single exp
if isinstance(ax, matplotlib.axes.SubplotBase):
    ax = [ax]

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

    sns.lineplot(ax=ax[i], data=exp_log, x='epoch', y='val_acc1', label="val", marker="o")
    sns.lineplot(ax=ax[i], data=exp_log, x='epoch', y='train_acc1', label="train", marker="o")

save_to_pdf(name="train.pdf")
