import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Set general plot properties.
sns.set()
sns.set_context("paper")
sns.set_color_codes("pastel")

# Use correct font types.
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Adjust as necessary.
sns.set_context({"figure.figsize": (16, 10)})

plt.style.use('grayscale')
plt.style.use('seaborn-white')
fig, ax = plt.subplots()
df = pd.DataFrame(...)
ax = sns.boxplot(data=df, color=sns.xkcd_rgb['light grey'])
ax = sns.swarmplot(data=df, color=sns.xkcd_rgb['dark grey'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel("y label", fontsize=30)
ax.set_xlabel('x label', fontsize=30)
ax.tick_params(labelsize=25)
plt.savefig("plot.pdf")
