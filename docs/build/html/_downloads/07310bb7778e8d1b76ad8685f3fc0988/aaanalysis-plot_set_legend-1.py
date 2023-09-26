import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
data = {'Classes': ['A', 'B', 'C'], 'Values': [23, 27, 43]}
colors = aa.plot_get_clist()
aa.plot_settings()
sns.barplot(x='Classes', y='Values', data=data, palette=colors)
sns.despine()
dict_color = dict(zip(["Class A", "Class B", "Class C"], colors))
aa.plot_set_legend(dict_color=dict_color, ncol=3, x=0, y=1.1, handletextpad=0.4)
plt.tight_layout()
plt.show()
