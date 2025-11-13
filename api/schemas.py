from pydantic import BaseModel
from typing import Optional, Literal

class JobCreate(BaseModel):
    kind: Literal["image","video"]

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued","started","finished","failed"]
    verdict: Optional[Literal["real","fake"]] = None
    prob_fake: Optional[float] = None
    message: Optional[str] = None
