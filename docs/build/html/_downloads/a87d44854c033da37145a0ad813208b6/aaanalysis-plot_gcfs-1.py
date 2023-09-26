import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
colors = aa.plot_get_clist()
aa.plot_settings()
sns.barplot(y='Classes', x='Values', data=data, palette=colors)
sns.despine()
plt.title("Big Title (+4 bigger than rest)", size=aa.plot_gcfs()+4)
plt.tight_layout()
plt.show()
