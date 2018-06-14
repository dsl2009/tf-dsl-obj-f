from multiprocessing import Process,Queue
from data_set.data_gen import get_batch_inception
from config.cfg import config_instace as config
import time
gen = get_batch_inception(config.batch_size,image_size=config.image_size[0])

def new_get_data(quene):
    while True:
        org_im, box, label,r_match,r_box = next(gen)
        quene.put([org_im, box, label,r_match,r_box])

numT = 4
q = Queue(numT)
ps = []
for p in range(numT):
    ps.append(Process(target=new_get_data,args=(q,)))
for pd in ps:
    pd.start()

