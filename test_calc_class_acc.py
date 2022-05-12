import torch

labelTensor = torch.tensor([27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 30,
        30, 30, 30,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
predTensor = torch.tensor([ 8,  5,  5,  2,  5,  8,  4,  5,  5,  5,  5,  4,  5,  5,  4,  5,  4,  5,
         5,  5,  5,  5,  3,  4,  5,  4,  4,  5,  5,  4,  5,  5,  0,  8,  5,  8,
         8,  4,  2,  2,  0,  8,  0,  5,  0,  5,  8,  8,  4,  2,  2,  8,  0,  4,
         8,  0,  8,  0,  5,  8,  8, 12,  4,  4])

print(labelTensor)
print(predTensor)

labelTotoalG = dict()
labelCorrectG = dict()

def updateClassCorrectCount(labelTotoal, labelCorrect):
    index = 0
    for v in labelTensor:
        label = v.item()
        if label in labelTotoal:
            labelTotoal[label] += 1
        else:
            labelTotoal[label] = 1

        if (label == predTensor[index].item()):
            if label in labelCorrect:
                labelCorrect[label] += 1
            else:
                labelCorrect[label] = 1
        index += 1

def printLabelAcc(labelTotoal, labelCorrect):
    for label in labelTotoal:
        correctCount = 0
        if label in labelCorrect:
            correctCount = labelCorrect[label]
        acc = correctCount / labelTotoal[label]
        print(f"label {label}, acc {acc}")

updateClassCorrectCount(labelTotoalG, labelCorrectG)
printLabelAcc(labelTotoalG, labelCorrectG)