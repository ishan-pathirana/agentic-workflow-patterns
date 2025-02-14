from typing import Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)
model = 'deepseek-r1:1.5b'

# Define Data Models

class AssistantRequestType(BaseModel):
    """Router LLM  call: Determine the type of assistant request"""

    request_type: Literal['light_config', 
                          'door_config', 
                          'entertainment_config',
                          'other'] = Field(
                              description='Type of assistant request'
                          )
    confidence_score: float = Field(
        description='Confidence score of the request type selection between 0 and 1'
    )
    description: str = Field(
        description='Cleand request text'
    )

class LightConfigDetails(BaseModel):
    """Details of a light configuration change"""

    place: str = Field(
        description='Place of the house where the light config change should be happened'
    )
    light_type: Literal['warm', 'cool'] = Field(
        description='Type of the light'
    )

class DoorConfigDetails(BaseModel):
    """Details of a door configuration change"""

    place: str = Field(
        description='Place of the house where the door config change should be happened'
    ),
    action: Literal['lock', 'unlock'] = Field(
        description='Action to be performed on the door lock'
    )

class EntertainmentConfigChange(BaseModel):
    """Details of a entertainment system configuration change"""

    action: Literal['play', 'stop', 'pause'] = Field(
        description='Action to be performed on the entertainment system'
    )
    genre: Optional[str]  = Field(
        description='Requsted genre to be played'
    )

class AssistantResponse(BaseModel):
    """Final response format"""

    status: str = Field(
        description='Whether the configuration chage is successful'
    )
    message: str = Field(
        description='User-friendly message'
    )

# Define routing and processing functions 

def route_agent_request(user_input: str) -> AssistantRequestType:
    """Router LLM call to determine the type of assistant request"""
    logger.info("Routing request")

    response = client.beta.chat.completions.parse(
        messages=[
            {
                'role': 'system',
                'content': '''Determine if this request is related to light configuration event or 
                            door configuration or entertainment configuration or other request'''    
            },
            {
                'role': 'user',
                'content': user_input
            }
        ],
        model=model,
        temperature=0,
        response_format=AssistantRequestType
    )

    result = response.choices[0].message.parsed
    logger.info(f'Request routed as: {result.request_type} with confidence: {result.confidence_score}')

    return result

def handle_light_config(description: str) -> LightConfigDetails:
    """LLM call to handle light configuraion change"""
    logger.info('Processing light configuraion change')

    response = client.beta.chat.completions.parse(
        messages=[
            {
                'role': 'system',
                'content': 'Extract the details for light configuration change'
            },
            {
                'role': 'user',
                'content': description
            }
        ],
        model=model,
        temperature=0,
        response_format=LightConfigDetails
    )

    result = response.choices[0].message.parsed
    logger.info(f'Light configuraion: {result.model_dump_json(indent=2)}')

    # Create response
    return AssistantResponse(
        status='success',
        Message=f'Light configuration change on {result.place} to {result.light_type}'
    )