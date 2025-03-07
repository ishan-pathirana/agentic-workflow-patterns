from ollama import chat, generate
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
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

class EventDetails(BaseModel):
    """Parse event details"""

    name: str = Field(description="Name of the event")
    description: str = Field(description="Description of the purpose of the event")
    date: str = Field(description="Date and time of the event. Use ISO 8601 to format this value")
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")    

class EventConfirmation(BaseModel):
    """Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
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
                'content': f'''{date_context} Analyze if the text describes a calender event and provide a confidence score between 0 and 1 about the decision.'''
            },
            {
                'role': 'user',
                'content': user_input
            }
        ],
        model='deepseek-r1:1.5b',
        options={'temperature': 0},
        format=EventValidation.model_json_schema()
    )

    event_validation = EventValidation.model_validate_json(response.message.content)
    logger.info(
        f"Validation complete - Is calender event: {event_validation.is_calender_event}, confidence: {event_validation.confidence_score:.2f}"
    )
    return event_validation

def extract_event_details(description: str) -> EventDetails:
    """Second LLM call to extract event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}"

    response = chat(
        messages=[
            {
                'role': 'system',
                'content': f'{date_context} Extract detailed event information. When dates referece "next tuesday" or similar relative details, use today as reference and calculate the date. Return response as JSON'
            },
            {
                'role': 'user',
                'content': description
            }
        ],
        model='deepseek-r1:1.5b',
        options={'temperature': 0},
        format=EventDetails.model_json_schema()
    )

    event_details = EventDetails.model_validate_json(response.message.content)
    return event_details

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    response = chat(
        messages=[
            {
                'role': 'system',
                'content': '''Generate a natural language calendar event add confirmation message for the event in a friendly tone. Include meeting name,
                            description, date and participants. Don't include any other information. Don't be creative.'''
            },
            {
                'role': 'user',
                'content': str(event_details.model_dump())
            }
        ],
        model='deepseek-r1:1.5b',
        options={'temperature': 0},
        format=EventConfirmation.model_json_schema(),
        keep_alive='0m'
    )

    confirmation = EventConfirmation.model_validate_json(response.message.content)
    logger.info("Confirmation message generated successfully")
    return confirmation

# Chaining prompts

def proces_calender_request(user_input: str) -> Optional[EventConfirmation]:
    """Chained LLM prompts with checks"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Validate request
    validation = validate_event(user_input=user_input)

    # Gate check: Verify for a calendar event with sufficient confidence
    if (not validation.is_calender_event) or (validation.confidence_score < 0.7):
        logger.warning(
            f"Validation failed - is_calendar_event: {validation.is_calender_event}, confindence: {validation.confidence_score}"
        )
        return None
    
    logger.info("Validation passed. proceeding wit event processing")

    # Second LLM call: Extract event information
    event_details = extract_event_details(description=user_input)

    # Third LLM call: Generate confirmation message
    confirmation = generate_confirmation(event_details=event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation

# Test

# Valid input

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."
result = proces_calender_request(user_input=user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
else:
    print("This doesn't appear to be a calendar event request.")

# Invalid input 

# user_input = "Generate a poem about roses"
# result = proces_calender_request(user_input=user_input)
# if result:
#     print(f"Confirmation: {result.confirmation_message}")
# else:
#     print("This doesn't appear to be a calendar event request.")