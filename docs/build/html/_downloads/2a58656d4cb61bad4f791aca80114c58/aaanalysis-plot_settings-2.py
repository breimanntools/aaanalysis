import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
colors = aa.plot_get_clist()
aa.plot_settings()
sns.barplot(data=data, x='Classes', y='Values', palette=colors, hue="Classes", legend=False)
sns.despine()
plt.title("Adjusted")
plt.tight_layout()
plt.show()
