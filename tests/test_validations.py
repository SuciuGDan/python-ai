import pytest
from lesson_03_data_validation import UserValidator
from pydantic import ValidationError


def test_simple_case_1():
    var1 = 10
    assert var1 <= 20
    assert var1 == 10


def test_user_validation():
    received_user = {
        "name": "Nicu",
        "age": 30,
        "nationality": "Romanian",
        "address": {
            "city": "Cluj",
            "street": "Principala",
        }
    }

    user = UserValidator.model_validate(received_user, strict=True)

    assert isinstance(user, UserValidator)
    assert len(user.name) < 10
    assert len(user.name) >= 2
    assert isinstance(user.name, str)
    assert isinstance(user.age, int)
    assert isinstance(user.nationality, str)


def test_invalid_user():
    user_data = {
        "name": 10,
        "age": 25,
        "nationality": "Romanian",
        "address": {
            "city": "Cluj",
            "street": "Principala",
        }
    }

    with pytest.raises(ValidationError):
        UserValidator.model_validate(user_data, strict=True)

@pytest.mark.parametrize("age",[-1, 13, 500, 100])
def test_user_age_validation(age):
  with pytest.raises(ValidationError):
    received_user = {
        "name": "Vicentiu",
        "age": 25,
        "nationality": "Romanian",
        "address": {
            "city": "Cluj",
            "street": "Principala",
        }
    }
    user = UserValidator.model_validate(received_user, strict=True)