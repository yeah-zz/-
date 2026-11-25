from native_fine_faint_20261101 import *
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
for ident in ['T01','T06','T17','T19']:
    im=Image.open(BASE/'originals'/(ident+'.png')).convert('RGB');d=ImageDraw.Draw(im)
    for x in range(0,1254,100):d.line([(x,0),(x,1253)],fill=(220,100,100),width=1);d.text((x+2,4),str(x),font=font,fill='red')
    for y in range(100,1254,100):d.line([(0,y),(1253,y)],fill=(220,100,100),width=1);d.text((4,y+2),str(y),font=font,fill='red')
    im.save(ROOT/(ident+'_grid.jpg'),quality=96)
