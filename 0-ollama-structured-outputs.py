from ollama import chat
from pydantic import BaseModel

class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]


response = chat(
    messages=[
        {
            'role': 'user',
            'content': 'tell me about canada'
        }
    ],
    model='deepseek-r1:1.5b',
    format=Country.model_json_schema()
)

country = Country.model_validate_json(response.message.content)
print(country)