class Student():
    
    def __init__(self, name=None, age=None, city=None):
        self.name = name
        self.age = age
        self.city = city
        self.greeting = f"My name is {self.name} and I am {self.age} years old"
        self.hello_world()
    
    
    def hello_world(self):
        print("Hello World")
        print(self.greeting)
        return None
    
    
    @staticmethod
    def goodbye_cruel_world():
        print("toodles!!!")
        return None