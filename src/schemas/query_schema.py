from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TravelQuerySchema(BaseModel):
    destination_name: Optional[str] = Field(
        default=None, 
        description="Target destination, city, district, or state name (e.g., 'Goa', 'North Goa', 'Rajasthan')."
    )
    region: Optional[str] = Field(
        default=None, 
        description="Geographic region if specified (e.g., 'West India', 'North India', 'South India')."
    )
    trip_types: List[str] = Field(
        default_factory=list, 
        description="List of requested trip categories, e.g., ['Beach', 'Cultural', 'Food', 'Nightlife', 'Wellness', 'Photography']."
    )
    ideal_for: Optional[str] = Field(
        default=None, 
        description="Traveler group archetype. Must match one of: 'Solo', 'Couple', 'Family_with_kids', 'Family_with_elderly', 'Friends'."
    )
    budget_tier: Optional[str] = Field(
        default="budget", 
        description="Budget preference category: 'budget', 'mid_range', or 'luxury'."
    )
    max_daily_budget_inr: Optional[int] = Field(
        default=None, 
        description="Explicit cap on daily total expenditure in INR per person."
    )
    season: Optional[str] = Field(
        default=None, 
        description="Season of travel, e.g., 'Winter', 'Summer', 'Monsoon', 'Post-Monsoon'."
    )
    duration_days: Optional[int] = Field(
        default=None, 
        description="Planned duration of the trip in days."
    )
    vibe_description: Optional[Union[str, List[str]]] = Field(
        default=None, 
        description="Semantic description or specific interest keywords for vector similarity matching."
    )

    @field_validator("vibe_description", mode="before")
    @classmethod
    def convert_vibe_list_to_str(cls, v):
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return v