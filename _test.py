import numpy as np
import pandas as pd
import itertools

import scipy.sparse.linalg
from scipy.sparse.linalg import svds

class Dog() :
    
    def sound(self) :
        print('Wo!')

class Cat() :
    
    def sound(self) :
        print('Meow!')

class Pets(Dog, Cat) :

    def call(self) :
        print(type(super(Dog, self)).__mro__)
        super().sound()
        super(Dog, self).sound()

mod = Pets()
mod.call()