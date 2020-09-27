import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import argparse

from shrinkbench.plot import df_from_results, plot_df, save_to_pdf
sns.set_theme(style="darkgrid")

parser = argparse.ArgumentParser(description="Monitor scripts for shrinkbench results")
parser.add_argument('-g', '--glob', default="*")
parser.add_argument('-p', '--path', default="./results")
args = parser.parse_args()


df, logs, params = df_from_results(f'{args.path}', glob=args.glob)

fig, ax = plt.subplots(len(logs), figsize=(5, 10), sharex=True)

# If there exists only a single exp
if isinstance(ax, matplotlib.axes.SubplotBase):
    ax = [ax]

max_epoch = []
sorted_results = sorted(zip(logs, params), key=lambda x: x[1]["pruning_kwargs"]["scheduler_args"]["n"])

for i, (exp_log, exp_param) in enumerate(sorted_results):
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
    if "weight_reset" in args and args["weight_reset"]:
        print("weight_reset: {}".format(args["weight_reset"]))
    else:
        print("weight_reset: False")

    sns.lineplot(ax=ax[i], data=exp_log, x='epoch', y='val_acc1', label="val", marker="o")
    sns.lineplot(ax=ax[i], data=exp_log, x='epoch', y='train_acc1', label="train", marker="o")
    max_epoch.append(max(exp_log["epoch"]))

for i in range(len(logs)):
    ax[i].set_xlim([0, max(max_epoch)])
    ax[i].set_ylim([0.8, 1])

plt.gcf().subplots_adjust(top=0.95)

save_to_pdf(name="train.pdf")
