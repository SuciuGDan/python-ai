import json

from pydantic import Strict

from lesson_03_data_validation import UserValidator

with open("user_data.json", 'r') as f:
    data = json.load(f)
    validated_user = UserValidator.model_validate(data, strict=True)
    print(validated_user)
