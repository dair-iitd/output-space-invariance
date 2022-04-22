import three;
currentprojection=perspective(20,20,20);
settings.prc = false;
settings.render = 0;
size(10cm);
int[] f={02,00,12,00,00,00,00,00,00,00,04,00,00,00,00,00,09,07,11,00,00,03,00,00,00,00,00,00,10,07,00,00,00,00,00,03,00,01,00,00,16,00,00,00,00,00,06,00,00,06,00,16,00,00,00,00,00,01,07,00,11,00,03,00,00,00,00,00,00,10,00,00,00,12,00,07,16,05,00,00,08,00,01,05,00,00,00,10,00,00,00,14,00,00,00,00,00,00,09,12,03,00,04,00,00,11,00,13,02,12,00,00,00,00,08,00,04,15,00,16,15,14,03,00,00,00,00,07,00,15,00,10,00,00,00,00,00,00,00,00,06,00,05,12,00,14,06,08,00,00,00,00,00,02,00,00,00,00,13,00,00,00,04,02,00,00,00,06,00,01,00,00,00,03,00,00,00,00,09,00,05,00,00,00,00,08,00,04,00,14,00,00};
path3 cell_vx = (0,0,0)--(1,0,0)--(1,0,1)--(0,0,1)--cycle;
path3 cell_vy = (0,0,0)--(0,1,0)--(0,1,1)--(0,0,1)--cycle;
path3 cell_h = (0,0,0)--(0,1,0)--(1,1,0)--(1,0,0)--cycle;
int block=0;
real pastell=4;
real r;
real g;
real b;
r=(1/pastell+1)*(pastell+sin(2*pi*0/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*0/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*0/16));
draw(surface(shift(floor((42-1)/8), 7.999, (42-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((45-1)/8), 7.999, (45-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((46-1)/8), 7.999, (46-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((50-1)/8), 7.999, (50-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((52-1)/8), 7.999, (52-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((53-1)/8), 7.999, (53-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((54-1)/8), 7.999, (54-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((55-1)/8), 7.999, (55-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((56-1)/8), 7.999, (56-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((58-1)/8), 7.999, (58-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((59-1)/8), 7.999, (59-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((60-1)/8), 7.999, (60-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((61-1)/8), 7.999, (61-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((62-1)/8), 7.999, (62-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((63-1)/8), 7.999, (63-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((64-1)/8), 7.999, (64-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*3/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*3/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*3/16));
draw(surface(shift(floor((1-1)/8), 7.999, (1-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((2-1)/8), 7.999, (2-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((3-1)/8), 7.999, (3-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((4-1)/8), 7.999, (4-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((5-1)/8), 7.999, (5-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((6-1)/8), 7.999, (6-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((7-1)/8), 7.999, (7-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((8-1)/8), 7.999, (8-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((11-1)/8), 7.999, (11-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((14-1)/8), 7.999, (14-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((15-1)/8), 7.999, (15-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((16-1)/8), 7.999, (16-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((22-1)/8), 7.999, (22-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((23-1)/8), 7.999, (23-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((30-1)/8), 7.999, (30-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((31-1)/8), 7.999, (31-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*6/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*6/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*6/16));
draw(surface(shift(floor((9-1)/8), 7.999, (9-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((17-1)/8), 7.999, (17-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((25-1)/8), 7.999, (25-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((26-1)/8), 7.999, (26-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((27-1)/8), 7.999, (27-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((28-1)/8), 7.999, (28-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((33-1)/8), 7.999, (33-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((34-1)/8), 7.999, (34-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((35-1)/8), 7.999, (35-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((36-1)/8), 7.999, (36-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((41-1)/8), 7.999, (41-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((43-1)/8), 7.999, (43-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((44-1)/8), 7.999, (44-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((49-1)/8), 7.999, (49-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((51-1)/8), 7.999, (51-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((57-1)/8), 7.999, (57-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*9/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*9/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*9/16));
draw(surface(shift(floor((10-1)/8), 7.999, (10-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((12-1)/8), 7.999, (12-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((13-1)/8), 7.999, (13-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((18-1)/8), 7.999, (18-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((19-1)/8), 7.999, (19-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((20-1)/8), 7.999, (20-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((21-1)/8), 7.999, (21-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((24-1)/8), 7.999, (24-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((29-1)/8), 7.999, (29-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((32-1)/8), 7.999, (32-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((37-1)/8), 7.999, (37-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((38-1)/8), 7.999, (38-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((39-1)/8), 7.999, (39-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((40-1)/8), 7.999, (40-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((47-1)/8), 7.999, (47-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((48-1)/8), 7.999, (48-1)%8)*cell_vx), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*1/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*1/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*1/16));
draw(surface(shift(7.999, floor((72-65)/8), (72-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((78-65)/8), (78-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((79-65)/8), (79-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((80-65)/8), (80-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((86-65)/8), (86-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((87-65)/8), (87-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((88-65)/8), (88-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((95-65)/8), (95-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((96-65)/8), (96-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((103-65)/8), (103-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((104-65)/8), (104-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((112-65)/8), (112-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((120-65)/8), (120-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((126-65)/8), (126-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((127-65)/8), (127-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((128-65)/8), (128-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*4/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*4/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*4/16));
draw(surface(shift(7.999, floor((94-65)/8), (94-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((102-65)/8), (102-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((107-65)/8), (107-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((109-65)/8), (109-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((110-65)/8), (110-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((111-65)/8), (111-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((114-65)/8), (114-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((115-65)/8), (115-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((116-65)/8), (116-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((117-65)/8), (117-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((118-65)/8), (118-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((119-65)/8), (119-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((122-65)/8), (122-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((123-65)/8), (123-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((124-65)/8), (124-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((125-65)/8), (125-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*7/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*7/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*7/16));
draw(surface(shift(7.999, floor((73-65)/8), (73-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((74-65)/8), (74-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((75-65)/8), (75-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((76-65)/8), (76-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((81-65)/8), (81-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((82-65)/8), (82-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((83-65)/8), (83-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((84-65)/8), (84-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((89-65)/8), (89-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((90-65)/8), (90-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((97-65)/8), (97-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((98-65)/8), (98-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((105-65)/8), (105-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((106-65)/8), (106-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((113-65)/8), (113-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((121-65)/8), (121-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*10/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*10/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*10/16));
draw(surface(shift(7.999, floor((65-65)/8), (65-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((66-65)/8), (66-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((67-65)/8), (67-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((68-65)/8), (68-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((69-65)/8), (69-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((70-65)/8), (70-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((71-65)/8), (71-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((77-65)/8), (77-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((85-65)/8), (85-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((91-65)/8), (91-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((92-65)/8), (92-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((93-65)/8), (93-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((99-65)/8), (99-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((100-65)/8), (100-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((101-65)/8), (101-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
draw(surface(shift(7.999, floor((108-65)/8), (108-65)%8)*cell_vy), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*2/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*2/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*2/16));
draw(surface(shift(floor((145-129)/8), (145-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((146-129)/8), (146-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((153-129)/8), (153-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((154-129)/8), (154-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((155-129)/8), (155-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((161-129)/8), (161-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((162-129)/8), (162-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((163-129)/8), (163-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((169-129)/8), (169-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((170-129)/8), (170-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((177-129)/8), (177-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((178-129)/8), (178-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((179-129)/8), (179-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((180-129)/8), (180-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((185-129)/8), (185-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((186-129)/8), (186-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*5/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*5/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*5/16));
draw(surface(shift(floor((168-129)/8), (168-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((171-129)/8), (171-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((172-129)/8), (172-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((173-129)/8), (173-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((175-129)/8), (175-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((176-129)/8), (176-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((181-129)/8), (181-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((182-129)/8), (182-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((183-129)/8), (183-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((184-129)/8), (184-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((187-129)/8), (187-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((188-129)/8), (188-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((189-129)/8), (189-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((190-129)/8), (190-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((191-129)/8), (191-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((192-129)/8), (192-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*8/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*8/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*8/16));
draw(surface(shift(floor((129-129)/8), (129-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((130-129)/8), (130-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((131-129)/8), (131-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((132-129)/8), (132-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((133-129)/8), (133-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((137-129)/8), (137-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((138-129)/8), (138-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((139-129)/8), (139-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((140-129)/8), (140-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((141-129)/8), (141-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((143-129)/8), (143-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((149-129)/8), (149-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((150-129)/8), (150-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((151-129)/8), (151-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((157-129)/8), (157-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((158-129)/8), (158-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
r=(1/pastell+1)*(pastell+sin(2*pi*11/16));
g=(1/pastell+1)*(pastell+sin(2*pi/3+2*pi*11/16));
b=(1/pastell+1)*(pastell+sin(4*pi/3+2*pi*11/16));
draw(surface(shift(floor((134-129)/8), (134-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((135-129)/8), (135-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((136-129)/8), (136-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((142-129)/8), (142-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((144-129)/8), (144-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((147-129)/8), (147-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((148-129)/8), (148-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((152-129)/8), (152-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((156-129)/8), (156-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((159-129)/8), (159-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((160-129)/8), (160-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((164-129)/8), (164-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((165-129)/8), (165-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((166-129)/8), (166-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((167-129)/8), (167-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
draw(surface(shift(floor((174-129)/8), (174-129)%8, 7.999)*cell_h), rgb(r,g,b),light=nolight);
path3 g;
g=(0, 8, 0) -- (0, 8, 8) -- (0, 0, 8) -- (8, 0, 8) -- (8, 0, 0) -- (8, 8, 0) -- cycle;
draw(g, black+linewidth(2pt));
g=(8, 8, 0) -- (8, 8, 8) -- (0, 8, 8);
draw(g, black+linewidth(2pt));
g=(8, 8, 8) -- (8, 0, 8);
draw(g, black+linewidth(2pt));
// vertical plane at y=8
real[] A={1.000000e+00,5.000000e-01, 1.000000e+00,1.500000e+00, 1.000000e+00,3.500000e+00, 1.000000e+00,4.500000e+00, 1.500000e+00,1.000000e+00, 1.500000e+00,2.000000e+00, 1.500000e+00,3.000000e+00, 2.000000e+00,2.500000e+00, 1.500000e+00,5.000000e+00, 2.000000e+00,7.500000e+00, 2.500000e+00,1.000000e+00, 3.000000e+00,1.500000e+00, 3.000000e+00,2.500000e+00, 3.000000e+00,3.500000e+00, 2.500000e+00,5.000000e+00, 2.500000e+00,7.000000e+00, 3.500000e+00,4.000000e+00, 3.500000e+00,5.000000e+00, 4.000000e+00,5.500000e+00, 3.500000e+00,7.000000e+00, 4.000000e+00,6.500000e+00, 5.000000e+00,1.500000e+00, 4.500000e+00,4.000000e+00, 5.000000e+00,4.500000e+00, 5.000000e+00,5.500000e+00, 5.500000e+00,1.000000e+00, 5.500000e+00,2.000000e+00, 5.500000e+00,4.000000e+00, 6.000000e+00,3.500000e+00, 5.500000e+00,6.000000e+00, 6.000000e+00,6.500000e+00, 6.000000e+00,7.500000e+00, 6.500000e+00,1.000000e+00, 6.500000e+00,2.000000e+00, 6.500000e+00,3.000000e+00, 7.000000e+00,2.500000e+00, 7.500000e+00,1.000000e+00, };
real x;
real y;
real z;
path3 g;
for (int i = 0; i < A.length/2; ++i) {
	x=A[2*i];
	z=A[2*i+1];
	if (x-floor(x)>0.1) // horizontal line
		g=(floor(x),8,round(z)) -- (floor(x)+1,8,round(z));
	else
		g=(round(x), 8, floor(z)) -- (round(x), 8, floor(z)+1);
	draw(g, black+linewidth(2pt));
}
// vertical plane at x=8
real[] B={1.000000e+00,5.000000e-01, 1.000000e+00,1.500000e+00, 1.000000e+00,2.500000e+00, 1.000000e+00,3.500000e+00, 1.000000e+00,5.500000e+00, 5.000000e-01,7.000000e+00, 1.000000e+00,6.500000e+00, 1.500000e+00,4.000000e+00, 1.500000e+00,5.000000e+00, 3.000000e+00,2.500000e+00, 2.500000e+00,4.000000e+00, 3.000000e+00,3.500000e+00, 2.500000e+00,5.000000e+00, 3.000000e+00,5.500000e+00, 3.500000e+00,2.000000e+00, 3.500000e+00,5.000000e+00, 3.500000e+00,6.000000e+00, 4.500000e+00,2.000000e+00, 5.000000e+00,2.500000e+00, 4.500000e+00,5.000000e+00, 5.000000e+00,4.500000e+00, 4.500000e+00,6.000000e+00, 5.000000e+00,6.500000e+00, 5.500000e+00,2.000000e+00, 6.000000e+00,1.500000e+00, 5.500000e+00,3.000000e+00, 5.500000e+00,4.000000e+00, 6.000000e+00,3.500000e+00, 5.500000e+00,7.000000e+00, 6.500000e+00,1.000000e+00, 7.000000e+00,5.500000e+00, 6.500000e+00,7.000000e+00, 7.000000e+00,6.500000e+00, 7.500000e+00,1.000000e+00, 7.500000e+00,5.000000e+00, };
for (int i = 0; i < B.length/2; ++i) {
	y=B[2*i];
	z=B[2*i+1];
	if (y-floor(y)>0.1) // horizontal line
		g=(8, floor(y),round(z)) -- (8, floor(y)+1,round(z));
	else
		g=(8, round(y), floor(z)) -- (8, round(y), floor(z)+1);
	draw(g, black+linewidth(2pt));
}
// horizontal plane at z=8
real[] C={5.000000e-01,5.000000e+00, 1.000000e+00,6.500000e+00, 2.000000e+00,5.000000e-01, 2.000000e+00,1.500000e+00, 2.000000e+00,2.500000e+00, 2.000000e+00,3.500000e+00, 1.500000e+00,5.000000e+00, 1.500000e+00,6.000000e+00, 2.000000e+00,5.500000e+00, 1.500000e+00,7.000000e+00, 2.500000e+00,2.000000e+00, 3.000000e+00,2.500000e+00, 2.500000e+00,4.000000e+00, 2.500000e+00,7.000000e+00, 3.000000e+00,6.500000e+00, 3.500000e+00,3.000000e+00, 3.500000e+00,4.000000e+00, 4.000000e+00,4.500000e+00, 3.500000e+00,6.000000e+00, 4.000000e+00,5.500000e+00, 4.000000e+00,7.500000e+00, 4.500000e+00,3.000000e+00, 5.000000e+00,2.500000e+00, 5.000000e+00,3.500000e+00, 5.000000e+00,4.500000e+00, 4.500000e+00,7.000000e+00, 5.000000e+00,6.500000e+00, 5.500000e+00,2.000000e+00, 6.000000e+00,2.500000e+00, 6.000000e+00,3.500000e+00, 5.500000e+00,5.000000e+00, 5.500000e+00,6.000000e+00, 6.000000e+00,5.500000e+00, 7.000000e+00,2.500000e+00, 6.500000e+00,4.000000e+00, 7.000000e+00,3.500000e+00, 7.500000e+00,2.000000e+00, };
for (int i = 0; i < C.length/2; ++i) {
	x=C[2*i];
	y=C[2*i+1];
	if (x-floor(x)>0.1) // line parallel to y axis
		g=(floor(x),round(y),8) -- (floor(x)+1,round(y),8);
	else
		g=(round(x), floor(y), 8) -- (round(x), floor(y)+1, 8);
	draw(g, black+linewidth(2pt));
}
int k=0;
for (int s = 0;s<3;++s) {
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			if (s==0) 
			{
				draw(shift(i, 8, j)*cell_vx, black+linewidth(0.5pt));
				if (f[k]>0)
					label(XZ()*string(f[k]), p = fontsize(13pt), (i+0.5,8,j+0.5), Embedded);
			}
			if (s==1) 
			{
				draw(shift(8, i, j)*cell_vy, black+linewidth(0.5pt));
				if (f[k]>0)
					label(YZ()*string(f[k]),p = fontsize(13pt), (8, i+0.5,j+0.5), Embedded);
			}
			if (s==2) 
			{
				draw(shift(i, j,8)*cell_h, black+linewidth(0.5pt));
				if (f[k]>0) // there must be an easier way to make the numbers in the top plane like this
					label(XY()*string(f[k]),p = fontsize(13pt), (i+0.5,j+0.5,8), Embedded);
			}
			k=k+1;
		}
	}
}
