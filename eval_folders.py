import os
import evaluate_from_model

path = "binarization/msbin/models"

for file in os.listdir(path):

    model_folder = os.path.join(path, file)
    print("------------------------- " + model_folder + " -------------------------")
    # input("Press Enter to continue...")
    evaluate_from_model.evaluate_fnc(model_folder)
