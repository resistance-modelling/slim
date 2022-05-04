#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

labels = [
    "master (serial)",
    "MP + queues (1)",
    "MP - queue (load=1)",
    "MP - queue (load=3)",
    "MP - queue - barrier (load=1)",
    "MP - queue - barrier (load=3)",
]

sublabels = ["30d", "60d", "180d", "365d"]
bar_width = 0.2


values = np.array(
    [
        [3.0, 2.82, 2.02, 2.20],
        [2.03, 1.86, 0.91, 1.39],
        [2.43, 2.08, 1.08, 1.40],
        [2.50, 2.43, 1.0, 1.41],
        [4.9, 4.1, 3.74, 3.76],
        [3.0, 2.5, 2.4, 1.98],
    ]
)

fig, ax = plt.subplots()
for idx, sublabel in enumerate(sublabels):
    df = pd.Series(dict(zip(labels, values[:, idx].flatten())))
    ax.bar(np.arange(len(labels)) + idx * bar_width, df, bar_width, label=sublabel)

ax.set_ylabel("Iterations per second")
ax.set_xticks(np.arange(len(labels)) + 2 * bar_width)
ax.set_xticklabels(labels)  # why doesn't the first one render???

ax.legend()

plt.show()
