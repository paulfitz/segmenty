#!/usr/bin/env python

import math
import svgwrite
import sys

if len(sys.argv) == 2:
    mode = sys.argv[1]
    assert mode in ["full", "mask", "vertical", "horizontal", "middle"]

svg = svgwrite.Drawing(size=("850px", "1100px"))

#  850
# 1100
# 80 * 9 = 720
# 80 + 50 => 130 / 2 => 65
# 65 - 80/2 = 25

xcenter = 850 / 2
ycenter = 1100 / 2

dx = 85
dy = 100
r0 = 16
r2 = 32
s2 = 8

cx = 9
cy = 11

if mode == "mask":
    svg.add(svg.circle(center=(xcenter, ycenter),
                       r=ycenter * 3,
                       fill='black',
                       stroke='black',
                       stroke_width=s2))

for ix in range(0, cx):
    for iy in range(0, cy):
        x = ix * dx + (850 - (cx - 1) * dx) // 2
        y = iy * dy + dy // 2
        if (ix + iy) % 2 == 0:
            if ix % 2 == 0 and iy % 2 == 0:
                xx = x - xcenter
                yy = y - ycenter
                length = math.sqrt(xx * xx + yy * yy)
                if abs(length) > 0.001:
                    xx /= length
                    yy /= length
                d = 3 * r2
                if mode == "full":
                    if ix % 4 == 0:
                        fill = 'black'
                    else:
                        fill = 'white'
                    svg.add(svg.circle(center=(x, y),
                                       r=r2,
                                       fill=fill,
                                       stroke='black',
                                       stroke_width=s2))
                    svg.add(svg.line(start=(x, y),
                                     end=(x - xx * d, y - yy * d),
                                     stroke='black',
                                     stroke_width=s2 * 2))
                elif mode == "horizontal":
                    svg.add(svg.line(start=(x - 200, y),
                                     end=(x + 200, y),
                                     stroke='black',
                                     stroke_width=s2 * 4))
                elif mode == "vertical":
                    svg.add(svg.line(start=(x, y - 200),
                                     end=(x, y + 200),
                                     stroke='black',
                                     stroke_width=s2 * 4))
                elif mode == "middle":
                    svg.add(svg.circle(center=(xcenter, ycenter),
                                       r=r2,
                                       fill='black',
                                       stroke='black',
                                       stroke_width=s2))
                    
            
print(svg.tostring())
