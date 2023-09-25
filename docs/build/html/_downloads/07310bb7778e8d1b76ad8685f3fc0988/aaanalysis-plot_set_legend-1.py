import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
data = {'Classes': ['A', 'B', 'C'], 'Values': [23, 27, 43]}
colors = aa.plot_get_cmap(name="CAT", n_colors=3)
aa.plot_settings()
sns.barplot(x='Classes', y='Values', data=data, palette=colors)
sns.despine()
dict_color = dict(zip(["Class A", "Class B", "Class C"], colors))
aa.plot_set_legend(dict_color=dict_color, marker="-", lw=10, loc_out=True)
plt.tight_layout()
plt.show()
