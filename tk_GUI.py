dots = [[1,1], [2,1], [2,2], [1,2], [0.5,1.5]]
from tkinter import Canvas
c = Canvas(width=750, height=750)
c.pack()
out = []
for x,y in dots:
    out += [x*250, y*250]
c.create_polygon(*out, fill='#aaffff')#fill with any color html or name you want, like fill='blue'
c.update()
input()