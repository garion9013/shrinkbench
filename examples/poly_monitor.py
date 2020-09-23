import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse

from shrinkbench.plot import df_from_results, plot_df, save_to_pdf
sns.set_theme(style="darkgrid")

parser = argparse.ArgumentParser(description="Monitor scripts for shrinkbench results")
parser.add_argument('-g', '--glob', default="*")
parser.add_argument('-p', '--path', default="./results")
args = parser.parse_args()


df, logs, params = df_from_results(f'{args.path}', glob=args.glob)

fig, ax = plt.subplots(2, figsize=(7, 10))

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

    label = exp_param["train_kwargs"]["optim"]+"-"+"-".join([ str(i) for i in list(args['scheduler_args'].values()) ])
    sns.lineplot(ax=ax[0], data=exp_log, x='epoch', y='val_acc1', label=label, marker="o")
    sns.lineplot(ax=ax[1], data=exp_log, x='epoch', y='val_loss', label=label, marker="o")

# ax.legend(title="", ncol=1, loc="lower left", bbox_to_anchor=[0.5, 0], frameon=False)
# plt.gcf().subplots_adjust(right=0.8)
save_to_pdf(name="train.pdf")
