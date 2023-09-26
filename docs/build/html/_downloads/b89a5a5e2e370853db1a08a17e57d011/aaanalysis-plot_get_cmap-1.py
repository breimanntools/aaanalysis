import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
colors = aa.plot_get_cmap(name="CPP", n_colors=3)
data = {'Classes': ['Class A', 'Class B', 'Class C',], 'Values': [10, 23, 33]}
aa.plot_settings(no_ticks_x=True, font_scale=1.2)
sns.barplot(x='Classes', y='Values', data=data, palette=colors)
plt.show()
