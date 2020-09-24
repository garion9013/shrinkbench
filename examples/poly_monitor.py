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

n = len(logs)

sorted_results = sorted(zip(logs, params), key=lambda x: x[1]["pruning_kwargs"]["scheduler_args"]["n"])

sns.color_palette("Set2")
for i, (exp_log, exp_param) in enumerate(sorted_results):
    print("")
    exp_result = df.iloc[i]
    print('{}, sparsity = {}, acc1 = {}'.format(
        exp_result["strategy"],
        1-1/exp_result["real_compression"],
        exp_result["post_acc1"]
    ))
    args = exp_param["pruning_kwargs"]
    # print(exp_param["train_kwargs"])
    print(args["scheduler"], args["scheduler_args"])
    if "weight_reset" in args and args["weight_reset"]:
        print("weight_reset: {}".format(args["weight_reset"]))
        wgt_rst = "reset"
    else:
        print("weight_reset: False")
        wgt_rst = ""

    label = exp_param["train_kwargs"]["optim"]+"-"+ \
            wgt_rst+"-"+ \
            "-".join([ str(i) for i in list(args['scheduler_args'].values()) ])
    kwargs = {"markers":True, "dashes":False, "palette":"flare"}
    sns.lineplot(ax=ax[0], data=exp_log, x='epoch', y='val_acc1', label=label, **kwargs)
    sns.lineplot(ax=ax[1], data=exp_log, x='epoch', y='val_loss', label=label, **kwargs)

# ax.legend(title="", ncol=1, loc="lower left", bbox_to_anchor=[0.5, 0], frameon=False)
# plt.gcf().subplots_adjust(right=0.8)
save_to_pdf(name="train.pdf")
