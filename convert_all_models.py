import os
import sys

sys.path.append("..")

from swins import model_configs


def main():
    for model_name in model_configs.MODEL_MAP:
        if "in22k" in model_name:
            dataset = "in21k"
        else:
            dataset = "in1k"

        for i in range(2):
            command = f"python convert.py -m {model_name} -d {dataset}"
            if i == 1:
                command += " -pl"
            os.system(command)


if __name__ == "__main__":
    main()
