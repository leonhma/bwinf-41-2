from wkt import WKT

with open('beispieldaten/wenigerkrumm1.txt') as f:
    outposts = [tuple(map(float, line.split())) for line in f.readlines()]

wkt = WKT(outposts, max_no_improvement=100000)
print(wkt.solve())
