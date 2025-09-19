import os
import json

from matplotlib import pyplot as plt
from pprint import pprint

from utils import print_with_padding

def get_frequency_distribution(data: dict) -> dict:
    """
    Get the frequency distribution of the data

    Args:
        data (dict): Dictionary to take the frequency distribution of

    Returns:
        dict: A dictionary with the frequency distribution of the data
    """
    frequencies = {}
    for value in data.values():
        if value not in frequencies:
            frequencies[value] = 0
        frequencies[value] += 1
    return frequencies

def draw_histogram(data: dict, title: str, x_label: str = "# of images", y_label: str = "Number of elephants"):
    """
    Draw a histogram displaying how many elephants have a certain number of images

    Args:
        data (dict): A frequency distribution of the data
        title (str): The title of the histogram
        x_label (str): The label of the x-axis
        y_label (str): The label of the y-axis
    """
    # Create the histogram
    x_values = list(data.keys())
    y_values = list(data.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, alpha=0.7, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(y_values):
        plt.text(x_values[i], v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


sheldrick_paths = ["processing/sheldrick/cropped/certain", "processing/sheldrick/cropped/probable"]
image_paths = []
sheldrick_elephants = {}
for folder in sheldrick_paths:
    for elephant in os.listdir(folder):
        if not elephant.endswith(".jpg"):
            continue
        elephant_id = elephant.split("_")[0]
        if elephant_id not in sheldrick_elephants:
            sheldrick_elephants[elephant_id] = 1
        else:
            sheldrick_elephants[elephant_id] += 1

elpephants_path = ["processing/ELPephants/cropped/certain", "processing/ELPephants/cropped/probable"]
elpephants = {}
for folder in elpephants_path:
    for image in os.listdir(folder):
        if not image.endswith(".jpg"):
            continue
        elephant_id = image.split("_")[0]
        if elephant_id not in elpephants:
            elpephants[elephant_id] = 1
        else:
            elpephants[elephant_id] += 1


print_with_padding("Sheldrick elephants")
draw_histogram(get_frequency_distribution(sheldrick_elephants), "Sheldrick elephants")
print(len(sheldrick_elephants))

print_with_padding("ELPephants")
draw_histogram(get_frequency_distribution(elpephants), "ELPephants")
print(len(elpephants))

# Merge the two dictionaries
merged_elephants = {**sheldrick_elephants}#, **elpephants}

#print_with_padding("Merged elephants")
#draw_histogram(get_frequency_distribution(merged_elephants), "Merged elephants")

freqs = get_frequency_distribution(merged_elephants)
# draw_histogram(freqs, "Merged elephants")

cutoff = input("Enter a cutoff value: ")

cut = {k: v for k, v in merged_elephants.items() if v >= int(cutoff)}

print("Images:    ", sum(cut.values()))
print("Elephants: ", len(cut))

draw_histogram(get_frequency_distribution(cut), "Cutoff elephants")