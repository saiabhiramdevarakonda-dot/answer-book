from pydantic import BaseModel, Field


class MarkSplit(BaseModel):
    marks: float
    for_: str = Field(alias="for")


class CommonMistake(BaseModel):
    wrong: str
    why: str


class GeneratedSolution(BaseModel):
    answer: str
    steps: list[str]
    mark_split: list[MarkSplit]
    common_mistakes: list[CommonMistake]