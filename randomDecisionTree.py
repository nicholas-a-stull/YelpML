from random import random

def dTree(feature, remaining, list):
    for element in list:
        if feature < element:
            randomIndex = randint(0, len(remaining) - 1)
            newFeat = remaining[randomIndex]
            newRemaining = remaining
            for x in range (0, len(newRemaining)):
                if x == randomIndex:
                    newRemaining.remove(newRemaining[x])
            dTree(newFeat, newRemaining, list)

if __name__ == "__main__":
