from wkt import WKT
import matplotlib.pyplot as plt

from utils import sliding_window

with open('beispieldaten/wenigerkrumm1.txt') as f:
    outposts = [tuple(map(float, line.split())) for line in f.readlines()]

wkt = WKT(outposts, max_no_improvement=100)
solution = wkt.solve()

i = 0
for x, y in solution:
    plt.plot(x, y, 'bo')
    i+=1

print(f'plotting {i} points')

plt.pause(10)

for (x1, y1), (x2, y2) in sliding_window(solution, 2):
    plt.plot([x1, x2], [y1, y2], linestyle='--')

plt.show()
