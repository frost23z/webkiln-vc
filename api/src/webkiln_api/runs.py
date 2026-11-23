from typing import Literal
from uuid import uuid7

from fastapi import APIRouter, Request
from pydantic import BaseModel
from webkiln_agents.core.schemas import Input


class RunRecord(BaseModel):
    brief: Input
    status: Literal["created", "running", "completed", "failed"] = "created"


class RunResponse(BaseModel):
    thread_id: str
    status: str


router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", status_code=201)
async def create_run(brief: Input, request: Request) -> RunResponse:
    thread_id = str(uuid7())
    record = RunRecord(brief=brief)
    request.app.state.runs[thread_id] = record
    return RunResponse(thread_id=thread_id, status=record.status)
