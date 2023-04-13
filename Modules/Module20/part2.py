import numpy as np

class Squares:
    def __init__(self, N):
        self.N = N
    
    def __getitem__(self, idx):
        ret = 0
        ret = (idx + 1) ** 2
        return ret
        
    def __len__(self):
        return self.N


N = 10
s = Squares(10)
out = "."
for i in range(len(s)):
    out = out + "{}.".format(s[i])

print(out)