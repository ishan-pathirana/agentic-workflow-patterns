from ollama import chat
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Set logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define data models

class EventValidation(BaseModel):
    """Extract basic information from event"""
    description: str = Field(description='Basic description of the event')
    is_calender_event: bool = Field(
        description='Whether this text describes a calender event'
        )
    confidence_score: float = Field(
        description='Confidence score between 0 and 1'
    )
    
# Define funcitons

def validate_event(user_input: str) -> EventValidation:
    """First LLM call to determine if input is a calender event that can be created on a calender with name, place, date, recipients etc."""
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}"

    response = chat(
        messages=[
            {
                'role': 'system', 
                'content': f'{date_context} Analyze if the text describes a calender event and provide a confidence score between 0 and 1'
            },
            {
                'role': 'user',
                'content': user_input
            }
        ],
        model='deepseek-r1:1.5b',
        format=EventValidation.model_json_schema()
    )

    event_validation = EventValidation.model_validate_json(response.message.content)
    logger.info(
        f"Validation complete - Is calender event: {event_validation.is_calender_event}, confidence: {event_validation.confidence_score:.2f}"
    )
    return event_validation

# Test

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."
response = validate_event(user_input=user_input)
print(response)