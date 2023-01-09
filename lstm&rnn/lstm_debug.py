import re
import torch

########## YOUR SOLUTION HERE ##########
class Encoder:
    def __init__(self, input_alphabet):
        self.alphabet = input_alphabet
    def __call__(self, input_string):
        encoded_string = []
        for character in input_string:
            for i in range(len(self.alphabet)):
                if character == self.alphabet[i]:
                    encoded_string.append(i)
                    break
        return torch.tensor(encoded_string)
    
encoder = Encoder("abcdefghijklmnopqrstuvwxyz0123456789 .!?")
print(encoder("cat"))