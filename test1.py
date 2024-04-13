class A():
    def __init__(self , max = 20 ):
        self.max = max


class B (A):
    def calculate(self):
        print(self.max * 2)

b = B()
b.calculate()
