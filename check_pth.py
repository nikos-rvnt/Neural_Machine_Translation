
import os

path = '/media/nikos/Data/didak/HealthSign/implement/gls2eng_transformer/check/'

pths = os.listdir(path)
os.mkdir('checkare')
for chk in range( len( pths )):
    if str(2)+".pth" not in pths[chk]: 
        print(pths[chk])
        os.remove(path + pths[chk])

