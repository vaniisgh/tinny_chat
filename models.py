from pydantic import BaseModel, RootModel
from typing import List, Dict


class Questions(BaseModel):
    questions: List[str]


class Response(RootModel[Dict[str, str]]):
    pass
