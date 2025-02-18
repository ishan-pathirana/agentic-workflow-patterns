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
model = 'deepseek-r1:8b'

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
    )
    action: Literal['lock', 'unlock'] = Field(
        description='Action to be performed on the door lock'
    )

class EntertainmentConfigDetails(BaseModel):
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
        message=f'Light configuration change on {result.place} to {result.light_type}'
    )

def handle_door_config(description: str) -> DoorConfigDetails:
    """LLM call to handle door configuraion change"""
    logger.info('Processing door configuraion change')

    response = client.beta.chat.completions.parse(
        messages=[
            {
                'role': 'system',
                'content': 'Extract the details for door configuration change'
            },
            {
                'role': 'user',
                'content': description
            }
        ],
        model=model,
        temperature=0,
        response_format=DoorConfigDetails
    )

    result = response.choices[0].message.parsed
    logger.info(f'Door configuraion: {result.model_dump_json(indent=2)}')

    # Create response
    return AssistantResponse(
        status='success',
        message=f'Door configuration change on {result.place} to {result.action}'
    )

def handle_entertainment_config(description: str) -> EntertainmentConfigDetails:
    """LLM call to handle entertainment configuraion change"""
    logger.info('Processing entertainment configuraion change')

    response = client.beta.chat.completions.parse(
        messages=[
            {
                'role': 'system',
                'content': 'Extract the details for entertainment configuration change.'
            },
            {
                'role': 'user',
                'content': description
            }
        ],
        model=model,
        temperature=0,
        response_format=EntertainmentConfigDetails
    )

    result = response.choices[0].message.parsed
    logger.info(f'Entertainment configuraion: {result.model_dump_json(indent=2)}')

    # Create response
    return AssistantResponse(
        status='success',
        message=f'Entertainment configuration change to {result.action} {f'{result.genre}' if result.genre else ""}'
    )

def process_assistant_request(user_input: str) -> Optional[AssistantResponse]:
    """Main function implementing the routing workflow"""
    logger.info('Processing assistant request')

    # Route the request
    route_result = route_agent_request(user_input=user_input)

    # Check confidence threshold
    if route_result.confidence_score < 0.7:
        logger.warning(f'Low confidence score: {route_result.confidence_score}')
        return None
    
    # Route to appropriate handler
    if route_result.request_type == 'light_config':
        return handle_light_config(route_result.description)
    elif route_result.request_type == 'door_config':
        return handle_door_config(route_result.description)
    elif route_result.request_type == 'entertainment_config':
        return handle_entertainment_config(route_result.description)
    else:
        logger.warning("Request type is not supported")
        return None
    
# Testing

# Test with light configuration change request
user_input = 'Change bedroom light to cool'
result = process_assistant_request(user_input=user_input)
if result:
    print(f'Response: {result.message}')

# Test with door configuration change request
user_input = 'lock the front door'
result = process_assistant_request(user_input=user_input)
if result:
    print(f'Response: {result.message}')

# Test with entertainment configuration change request
user_input = 'play some jazz'
result = process_assistant_request(user_input=user_input)
if result:
    print(f'Response: {result.message}')