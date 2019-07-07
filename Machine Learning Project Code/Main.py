from DigitClassification_SimpleNN import SimpleNN
from DigitClassification_CNN import CNN
from DigitClassification_SVM import SVM
from DigitRecognition_SVM import Digit_Recognition

import sys
if sys.version[0]=="3": raw_input=input

print("=========================================")
print("Method Menu: ")
print("1    -   Simple Neural Network")
print("2    -   Convolutional Neural Network")
print("3    -   SVM")
print("4    -   SVM Digit Recognition")

method = raw_input("Please enter method number:")

if method=="1":
    simpleNN = SimpleNN()
    error = simpleNN.build_model()
    print(error)
elif method=="2":
    CNN = CNN()
    error = CNN.build_model()
    print(error)
elif method=="3":
    SVM = SVM()
    SVM.build_model()
    score = SVM.get_score()
    print(score)
elif method=="4":
    photo = raw_input("Please enter photo name:")
    digit_reco = Digit_Recognition(photo)
    digit_reco.recognize_digit()