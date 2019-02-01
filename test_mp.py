import torch
import torch.multiprocessing as mp
import time
import resource
import os

def pr(count, log_every, sleep_time,queue, can_exit, cuda):
    for i in range(count):
        if not (i+1) % log_every:
            print("Produce %s" % i)
            # print_open_fds()
            if sleep_time:
                time.sleep(sleep_time)
        a = torch.tensor([ [i] * 2 ])
        if cuda:
            a = a.cuda()
        queue.put(a)
        del a
    can_exit.wait()

def toss(id, count, log_every,q1,q2, can_exit):
    for i in range(count):
        if not (i+1) % log_every:
            print("Toss(%d) %s" % (id,i))
        x = q1.get()
        y = x.clone()
        q2.put(y)
    can_exit.wait()

def ca(count, log_every,queue, res, can_exit, cuda):

    acc = torch.tensor([ [0] * 2 ])
    if cuda:
        acc = acc.cuda()
    for i in range(count):
        if not (i+1) % log_every:
            print("Getting %s" % i)
        x = queue.get()
        acc += x
        # x = 0
    res.put(acc)
    can_exit.wait()







if __name__ == '__main__':

    mp.set_start_method('spawn')

    count = 10
    log_every= count // 10
    cuda = True

    def do_process(next_function, **kwargs):
        out_q = mp.Queue()
        ev = mp.Event()
        proc = mp.Process(**kwargs)
        proc.start()
        next_function(out_q)
        ev.set()
        proc.join()


    def produce(in_q, next_function):
        out_q = mp.Queue()
        ev = mp.Event()
        proc = mp.Process(target = pr, args=(count,log_every,0, out_q , ev, cuda))
        proc.start()
        next_function(out_q)
        ev.set()
        proc.join()

    def calc(in_q, next_function):
        out_q = mp.Queue()
        ev = mp.Event()
        proc = mp.Process(target = ca, args=(count,log_every, in_q, out_q , ev, cuda))
        proc.start()
        next_function(out_q)
        ev.set()
        proc.join()


    def done(in_q, next_function):
        x = in_q.get()
        y = x.clone()
        del x
        expected = ((count-1) * count) // 2
        print(y)
        print(expected)
        print(y.data[0][0] == expected)



    produce(None, lambda x: calc(x, lambda x: done(x, None)))

    # p1 = mp.Process(target = pr, args=(count,log_every,0,q2,e1, cuda))
    # p2 = mp.Process(target = toss, args=(count,log_every,q1,q2,e2))
    # p3 = mp.Process(target = ca, args=(count,log_every,q2,q3,e3, cuda))

    # input("Press Enter to continue...")

    # p1.start()
    # p2.start()
    # p3.start()


    # e3.set()
    # p3.join()

    # e2.set()
    # p2.join()

    # e1.set()
    # p1.join()
