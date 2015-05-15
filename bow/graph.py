import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
ggcolors = plt.rcParams['axes.color_cycle']

df = pd.read_table('thresh.csv', sep=',', index_col=0)
df.columns = ['Development - Seen',
              'Development - Unseen',
              'Test - Seen',
              'Test - Unseen']

fig = plt.figure(figsize=(6, 9))
ax = plt.subplot()

ax.fill_between(
        range(15, 21),
        df['Development - Seen'][range(15,21)],
        color = '#d5d5d5')

df.plot(ylim = [0.55, 0.85],
        ax = ax,
        style = ['.-','.--','.-','.--'])

lines = ax.get_lines()
colors = ['#333333',
          '#333333',
          ggcolors[0],
          ggcolors[0]]
for i,l in enumerate(lines):
    l.set_color(colors[i])
        #colors =['r','k','r','k']) #['#000000','#666666','#000000','#666666'])
plt.legend(loc=4)
plt.title('Stop-Word Selection vs. Recognition Accuracy')
plt.xlabel('Count Threshold')
plt.ylabel('Metaphor Recognition Accuracy')
plt.tight_layout()
plt.savefig('stop-words.pdf')
plt.show()
