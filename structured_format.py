from pydantic import BaseModel
from openai import OpenAI
from typing import List, Optional


class QuestionSet(BaseModel):
    when: str
    where: str
    what: str
    who: str
    why: str
    how: str


class AnswersList(BaseModel):
    Answers: List[str]


class FactCheck(BaseModel):
    factuality: str
    factuality_score: int
    summary: str
    key_facts: List[str]


class SearchResultSummary(BaseModel):
    search_findings: str
    relationship_to_statement: str
    confidence_level: str

