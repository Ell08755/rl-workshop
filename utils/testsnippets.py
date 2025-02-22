# 试验一下numpy小函数的地方
import numpy as np
layout = [
    "S.......",
    "........",
    "...H....",
    ".....H..",
    "...H....",
    ".HH...H.",
    ".H..H.H.",
    "...H...G"
]
for row in layout:
    print(row)
m=[list(row) for row in layout]
print(m)
j=np.where(np.array(m)=='H')
j=np.vstack(j)
print(j)
print(j.T)
print([h for h in j.T])

