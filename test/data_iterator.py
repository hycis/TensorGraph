import tensorgraph as tg
import numpy as np
import time


def test_DataBlocks():
    X = np.random.rand(100000, 2000)
    with open('X.npy', 'wb') as f:
        np.save(f, X)

    db = tg.DataBlocks(['X.npy']*10, batchsize=32, allow_preload=True)
    t1 = time.time()
    count = 1
    for blk in db:
        print count
        count += 1
        for batch in blk:
            print sleep(0.1)
            pass
    print 'with preload time:', time.time() - t1

    db = tg.DataBlocks(['X.npy']*10, batchsize=32, allow_preload=False)
    t1 = time.time()
    count = 1
    for blk in db:
        print count
        count += 1
        for batch in blk:
            print sleep(0.1)
            pass
    print 'without preload time:', time.time() - t1


    db = tg.DataBlocks([('X.npy', 'X.npy'), ('X.npy', 'X.npy')], batchsize=32, allow_preload=False)
    for blk in db:
        print blk
        for batch in blk:
            print 'len batch:', len(batch)
            print 'batch1 size:', batch[0].shape
            print 'batch2 size:', batch[1].shape


if __name__ == '__main__':
    test_DataBlocks()
