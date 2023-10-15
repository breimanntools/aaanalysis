import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
data = {'Classes': ['A', 'B', 'C'], 'Values': [23, 27, 43]}
colors = aa.plot_get_clist()
aa.plot_settings()
sns.barplot(x='Classes', y='Values', data=data, palette=colors, hatch=["/", ".", "."], hue="Classes", legend=False)
sns.despine()
dict_color = {"Group 1": "black", "Group 2": "black"}
aa.plot_legend(dict_color=dict_color, ncol=2, y=1.1, hatch=["/", "."])
plt.tight_layout()
plt.show()
