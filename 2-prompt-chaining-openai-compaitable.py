from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Create an OpenAI client with ollama openai compaitable API
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
model = "deepseek-r1:1.5b"

# Define data models

class EventValidation(BaseModel):
    """Event validation definition"""

    is_calendar_event: bool = Field(
        description="Whether the text describes a calendar event"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1"
    )

class EventDetails(BaseModel):
    """Event Details"""

    name: str = Field(
        description="Name of the event"
    )
    date: datetime = Field(
        description="Date and time of the event. Use YYYY-MM-DD format"
    )
    duration_minutes: int = Field(
        description="Event duration in minutes"
    )
    participants: list[str] = Field(
        description="List of participants"
    )

class EventConfirmation(BaseModel):
    """Event creation confirmation message"""

    confirmation_message: str = Field(
        description="Natural Language confirmation message"
    )

# Define functions

def validate_event(user_input: str) -> EventValidation:
    """First LLM call to determine if the user input is a
    valid calender event"""
    logger.info("Starting event validation")
    logger.debug(f"Input text: {user_input}")

    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": "Analyze thorougly of tje text describes a calendar event."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        model=model,
        temperature=0,
        response_format=EventValidation
    )

    result = response.choices[0].message.parsed
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result

def extract_event(user_input: str) -> EventDetails:
    """Second LLM call to extract the event details"""

    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": f"""Extract detailed event information. When dates reference 'next tuesday' or similar
                            relative dates use today to calculate the meeting date in YYYY-MM-DD format. {date_context}"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        model=model,
        temperature=0,
        response_format=EventDetails
    )

    result = response.choices[0].message.parsed
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""

    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": "Generate a event creation confirmation message with event detail. Don't include anything other than event details"
            },
            {
                "role": "user",
                "content": str(event_details.model_dump())
            }
        ],
        model=model,
        temperature=0,
        response_format=EventConfirmation
    )

    result = response.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    return result

# Chain functions

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: validate the user input
    validation = validate_event(user_input=user_input)

    # Gate check: verify if the user input is a calendar event
    if (
        not validation.is_calendar_event or 
        validation.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {validation.is_calendar_event}, confidence: {validation.confidence_score}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")
    
    # Second LLM call: extract event details
    event_details = extract_event(user_input=user_input)

    # Third LLM call: generate confirmation
    confirmation = generate_confirmation(event_details=event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation

# Test

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

result = process_calendar_request(user_input=user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
else:
    print("This doesn't appear to be a calendar event request.")