import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
colors = aa.plot_get_clist(n_colors=3)
data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [10, 23, 33]}
aa.plot_settings()
sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes", legend=False)
plt.show()
