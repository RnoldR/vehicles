#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:58:38 2021

@author: arnold

Test virtual methods when prefixing a class
"""

class A():
    def __init__(self):
        print('A.__init__')
        
    def method(self, par1):
        print('A.method(', str(par1), ')')
        
class B(A):
    def __init__(self):
        super().__init__()
        print('B.__init__')
        
    def method(self, par1, par2):
        super().method(par1)
        print('B.method(', str(par1), ',', str(par2), ')')
        super().method('super again')
       
print('[Create instance of B]')        
b = B()

print('[Call B.method]')
b.method('parameter 1', 'parameter2')
