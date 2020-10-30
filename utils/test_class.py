class Person:
    def __init__(self, name = "megh"):
        self.name = name
        self.x = {'f':self.get_person_name}

    def _helper(self, x):
        return x

    def get_person_name(self):
        print(self)
        return self._helper(self.name)

p = Person()

print(p.get_person_name())