from structure_train_v20 import *
folder=ROOT/'paired_curriculum'/'pairs';ids=['basic_1_2','fupi_large_8_9','guatie_24_25','jiesuo_36_37','yudian_67_68'];board=Image.new('RGB',(1600,370*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,4),'新增配对监督抽查：红色新增，蓝色已有；最后一行为整组留出示范',font=FONT,fill='black')
for i,k in enumerate(ids):board.paste(Image.open(folder/(k+'_audit.jpg')),(0,35+i*370))
board.save(ROOT/'paired_curriculum'/'PAIRED_SELECTED_AUDIT.jpg',quality=95)
