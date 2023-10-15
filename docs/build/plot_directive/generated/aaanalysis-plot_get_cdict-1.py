import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
dict_color = aa.plot_get_cdict(name="DICT_COLOR")
data = {"Keys": list(dict_color.keys()), 'Values': [1] * len(dict_color) }
aa.plot_settings(weight_bold=False)
ax = sns.barplot(data=data, x="Values", y="Keys", palette=dict_color, legend=False)
ax.xaxis.set_visible(False)
sns.despine()
plt.tight_layout()
plt.show()
