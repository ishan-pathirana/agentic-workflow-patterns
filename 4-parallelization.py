from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os

# Configure logging
logging.basicConfig(
    level= logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
model = "deepseek-r1:8b"

class AssistantRequestValidation(BaseModel):
    """Validation details of the user input for the assistant"""

    is_assistant_request: bool = Field(
        description="Whether this is a valid assistant request."
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1"
    )

class SecurityCheck(BaseModel):
    """Check for prompt injection or system manipulation attempts"""

    is_safe: bool = Field(
        description="Whether the input appears safe"
    )
    risk_flags: list[str] = Field(
        description="List of potential security concerns"
    )

