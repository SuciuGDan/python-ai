import json



class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.nationality = "Romanian"

    def to_json(self):
        d1 = {
            "name":self.name,
            "age":self.age,
            "nationality":self.nationality
        }
        return json.dumps(d1)

    def say_hello(self):
        print(self.name + " says hi!")

    def __str__(self):
        return "" + self.name + " " + str(self.age) + ", " + self.nationality


dan = User("Dan",26)
dragos = User("Dragos",35)

# print(dragos)

# print(dan.name)

# print(dan.name,dan.age,dan.nationality)
if __name__ == "__main__":
    dan.say_hello()
    print(dan)

    print(dragos.to_json())

    # set - unordered list

    s1 = set([1,2,3])

    print(s1)

    d1 = {
        "name":"Debra"
    }

    print(d1["name"])

    # JSON

    json_text = '{"name":"Jason","age":"23"}'
    created_dict = json.loads(json_text)

    print(created_dict["name"])