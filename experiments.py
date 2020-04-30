import os
from main import start_process


def start(path):
    datastr = os.listdir(path)
    datastr.sort(reverse=False)
    train = True
    print(datastr)
    if 'test' in path:
        train = False
    print('Train: {}'.format(train))
    for i, name in enumerate(datastr):
        print(name)
        start_process(path, name, train=train)


paths = ['mot20/train/', 'mot20/test/']

for p in paths:
    start(p)
