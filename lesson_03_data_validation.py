from pydantic import BaseModel, ValidationError, Field
from lesson_02_classes_objects import User
from typing import Literal

class Address(BaseModel):
    city: str
    street: str

class UserValidator(BaseModel):
    name: str = Field(min_length=2, max_length=10)
    age: int = Field(ge=0, le=120)
    nationality: Literal["Romanian", "Moldovean"] = "Moldovean"
    external: bool | None = None
    address: Address


received_user = {
    "name": "Nicu",
    "age": 30,
    "nationality": "Romanian",
    "address":{
        "city": "Cluj",
        "street": "Romanian",
}
}

print("=======================Validations=======================")
try:
    validated_user = UserValidator.model_validate(received_user, strict=True)
    print(validated_user)
except ValidationError as e:
    print(e)
    print(e.errors)