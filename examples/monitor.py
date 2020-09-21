import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import argparse

parser = argparse.ArgumentParser(description="Monitor scripts for shrinkbench results")
parser.add_argument('-g', '--glob', default="*")

args = parser.parse_args()

from shrinkbench.plot import df_from_results, plot_df, save_to_pdf
sns.set_theme(style="darkgrid")


df, logs = df_from_results('results', glob=args.glob)
# plot_df(df, 'real_compression', 'pre_acc1', markers='strategy', fig=False, line='--', colors='strategy', suffix=' - pre')
# save_to_pdf(name="output.pdf")
# plt.close()

fig, ax = plt.subplots(len(logs), figsize=(5, 10))

# If there exists only a single exp
if isinstance(ax, matplotlib.axes.SubplotBase):
    ax = [ax]

for i, exp_log in enumerate(logs):
    exp_result = df.iloc[i]
    print(exp_result["strategy"], exp_result["real_compression"], exp_result["post_acc1"])
    print(exp_log)
    sns.lineplot(ax=ax[i], data=exp_log, x='steps', y='val_acc1', label="val", marker="o")
    sns.lineplot(ax=ax[i], data=exp_log, x='steps', y='train_acc1', label="train", marker="o")

save_to_pdf(name="train.pdf")
