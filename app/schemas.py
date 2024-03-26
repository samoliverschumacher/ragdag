from datetime import datetime

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """`user_query`: `str`,
    `metdata`: `dict`
    """

    user_query: str
    metadata: dict = {}


class CleansedRequest(GenerateRequest):
    """
    Users query has no sensitive information that might be leaked to other services.
    """

    text: str = Field(..., alias="user_query")
    user_id: str = None

    def __init__(self, **data):
        super().__init__(**data)


class GenerateResponse(BaseModel):
    generated_text: str


class PatchArticleItem(BaseModel):
    doc_id: str
    text: str | None
    created_by: str | None = None
    created_date: datetime | None = None


class PatchArticleRequest(BaseModel):
    items: list[PatchArticleItem]


class GetArticleRequest(BaseModel):
    doc_id: str


class GetArticleResponse(BaseModel):
    doc_id: str
    text: str | None
    created_by: str | None = None
    created_date: datetime | None = None
