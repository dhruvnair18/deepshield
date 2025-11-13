from redis import Redis
from rq import Queue
from rq.job import Job

from .schemas import JobStatus
from .model_runner import analyze_image, analyze_video



def enqueue_analysis(q: Queue, job_id: str, kind: str, path: str) -> Job:
    if kind == "image":
        return q.enqueue(analyze_image, job_id, path, job_id=job_id)
    else:
        return q.enqueue(analyze_video, job_id, path, job_id=job_id)

def get_job_result(redis: Redis, job_id: str) -> JobStatus:
    job = Job.fetch(job_id, connection=redis)
    if job.is_queued or job.get_status() == "queued":
        return JobStatus(job_id=job.id, status="queued")
    if job.get_status() == "started":
        return JobStatus(job_id=job.id, status="started")
    if job.get_status() == "failed":
        return JobStatus(job_id=job.id, status="failed", message=str(job.exc_info)[-4000:])
    # finished
    res = job.result or {}
    return JobStatus(
        job_id=job.id,
        status="finished",
        verdict=res.get("verdict"),
        prob_fake=res.get("prob_fake"),
        message=res.get("message"),
    )
