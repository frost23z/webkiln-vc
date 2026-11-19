from pydantic import BaseModel, ConfigDict, Field


class Input(BaseModel):
    brand_name: str = Field(min_length=2, max_length=80)
    prompt: str = Field(min_length=20, max_length=2000)
    niche: str = Field(min_length=3, max_length=80)

    model_config = ConfigDict(str_strip_whitespace=True)
