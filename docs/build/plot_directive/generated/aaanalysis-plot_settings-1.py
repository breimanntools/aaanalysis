import matplotlib.pyplot as plt
import seaborn as sns
import aaanalysis as aa
data = {'Classes': ['Class A', 'Class B', 'Class C'], 'Values': [23, 27, 43]}
sns.barplot(x='Classes', y='Values', data=data)
sns.despine()
plt.title("Seaborn default")
plt.tight_layout()
plt.show()
