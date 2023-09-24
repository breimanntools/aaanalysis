import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
colors = aa.plot_get_cmap(name="TAB", n_colors=4)
data = {'Classes': ['Class A', 'Class B', 'Class C', "Class D"], 'Values': [23, 27, 43, 38]}
aa.plot_settings(no_ticks_x=True, font_scale=1.2)
sns.barplot(x='Classes', y='Values', data=data, palette=colors)
plt.show()
