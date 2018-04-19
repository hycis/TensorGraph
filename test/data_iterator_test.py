import tensorgraphx as tg
import numpy as np
import time


def test_SimpleBlocks():
    X = np.random.rand(100, 200)
    with open('X.npy', 'wb') as f:
        np.save(f, X)

    db = tg.SimpleBlocks(['X.npy']*10, batchsize=32, allow_preload=True)
    t1 = time.time()
    count = 1
    for blk in db:
        print(count)
        count += 1
        for batch in blk:
            print(time.sleep(0.1))
            pass
    print('with preload time:', time.time() - t1)

    db = tg.SimpleBlocks(['X.npy']*10, batchsize=32, allow_preload=False)
    t1 = time.time()
    count = 1
    for blk in db:
        print(count)
        count += 1
        for batch in blk:
            print(time.sleep(0.1))
            pass
    print('without preload time:', time.time() - t1)


    db = tg.SimpleBlocks([('X.npy', 'X.npy'), ('X.npy', 'X.npy')], batchsize=32, allow_preload=False)
    for blk in db:
        print(blk)
        for batch in blk:
            print('len batch:', len(batch))
            print('batch1 size:', batch[0].shape)
            print('batch2 size:', batch[1].shape)


def test_DataBlocks():
    X = np.random.rand(100, 200)
    with open('X.npy', 'wb') as f:
        np.save(f, X)

    db = tg.DataBlocks(['X.npy']*10, batchsize=32, allow_preload=False)
    for train_blk, valid_blk in db:
        n_exp = 0
        pbar = tg.ProgressBar(len(train_blk))
        for batch in train_blk:
            n_exp += len(batch[0])
            time.sleep(0.05)
            pbar.update(n_exp)
        print()
        pbar = tg.ProgressBar(len(valid_blk))
        n_exp = 0
        for batch in valid_blk:
            n_exp += len(batch[0])
            time.sleep(0.05)
            pbar.update(n_exp)
        print()

if __name__ == '__main__':
    test_DataBlocks()
    test_SimpleBlocks()
