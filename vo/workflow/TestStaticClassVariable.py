from __future__ import print_function

class A(object):
    classVariable = None

    def __init__(self):
        pass
    
    def show_class_variable(self):
        if ( A.classVariable is None ):
            print( "classVariable is None" )
        else:
            print(A.classVariable)
    
    def change_class_variable(self, var):
        A.classVariable = var

if __name__ == "__main__":
    a = A()

    a.show_class_variable()

    aa = A()

    a.change_class_variable(1)

    aa.show_class_variable()


