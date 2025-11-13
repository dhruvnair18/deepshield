import os, sys
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

listen = ['analysis']
conn = Redis.from_url(os.getenv('REDIS_URL','redis://localhost:6379/0'))

if __name__ == '__main__':
    q = Queue('analysis', connection=conn)
    worker = SimpleWorker([q], connection=conn)
    worker.work(with_scheduler=False)
