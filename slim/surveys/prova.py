import ray
from ray.util.queue import Queue

ray.init()

@ray.remote
class F:
    def __init__(self, i):
        print(f"{i} was created")
        self.i = i

    def f(self, q1: Queue, q2: Queue):
        print(f"{self.i} was started")
        while True:
            x = q1.get()
            if x == 0:
                return
            print("Received x, computing...")
            q2.put(x*2)

n = 10
attempts = 5
q1 = [Queue(maxsize=1000) for _ in range(n)]
q2 = [Queue(maxsize=1000) for _ in range(n)]

actors = [F.remote(i) for i in range(n)]
ref = [actors[i].f.remote(q1[i], q2[i]) for i in range(n)]

for attempt in range(attempts):
    for j in range(n):
        q1[j].put(10)
    for j in range(n):
        x = q2[j].get()
        assert x == 20

for j in range(n):
    q1[j].put(0)

ray.get(ref) # gracefully close
