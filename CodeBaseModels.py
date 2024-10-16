from pydantic import BaseModel, Field

class CodeFile(BaseModel):
    FILENAME: str = Field(description="Name of the file that contains name of the code file")
    RAW_CODE: str = Field(description="The raw code for the file")

class CodeReview(BaseModel):
    REVIEW: str = Field(description="Review of the code")
    SCORE: int = Field(description="Score of the code")