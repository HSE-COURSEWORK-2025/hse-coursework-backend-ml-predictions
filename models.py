from pydantic import BaseModel
from typing import Optional

class SleepDisorderInput(BaseModel):
    gender: Optional[str] = 'Male'
    age: Optional[int] = 22
    sleep_duration_hours: float
    physical_activity_mins_daily: int
    bmi_category: str

    heart_rate: int
    daily_steps: int

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_gender
        yield cls.validate_bmi_category

    @staticmethod
    def validate_gender(value):
        allowed = {"Male", "Female"}
        if value not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return value

    @staticmethod
    def validate_bmi_category(value):
        allowed = {"Overweight", "Normal Weight", "Obese", "Normal"}
        if value not in allowed:
            raise ValueError(f"bmi_category must be one of {allowed}")
        return value


class SleepDisorderOutput(BaseModel):
    Insomnia: float
    Sleep_Apnea: float
    nan: float

