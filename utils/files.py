import os
import random
from .input import get_int, get_multiple_choice


def get_list_of_files(prompt: str, *, input_method: str | None = None) -> list[str]:
    """
    Gets a list of files from the user.

    Args:
        prompt (str): The prompt to display to the user. Do not include a colon or newline at the end.
        input_method (str): The input method to use. Defaults to None - user will be prompted to choose. Choices are "drag and drop" (or "1"), and "manual typing" (or "2").

    Returns:
        list[str]: The list of files entered by the user.
    """
    if input_method is None:
        input_method = get_multiple_choice("Which input method would you like to use?\n[1] Drag and Drop\n[2] Manual Typing\n", choices=["1", "2"])
    else:
        if input_method not in ["1", "2"]:
            if input_method == "drag and drop":
                input_method = "1"
            elif input_method == "manual typing":
                input_method = "2"
            else:
                raise ValueError("Invalid input method.")
    if input_method == "1":
        while True:
            print(prompt)
            print("Please drag and drop files from finder into the terminal.")
            files_input = input()
            try:
                files = files_input[1:-1].split("''")
                if all(os.path.isfile(file) for file in files):
                    return files
                else:
                    missing_files = [file for file in files if not os.path.isfile(file)]
                    print(f"The following files were not found: {missing_files}")
                    print("Please try again.")
            except Exception as e:
                print(f"Error parsing file paths: {e}")
                print("Please try again.")
    elif input_method == "2":
        return get_files_from_dir(prompt, randomize=False)


def get_files_from_dir(prompt: str, base_path: str | None = None, randomize: bool = False) -> list[str]:
    """
    Gets a list of files from a directory.

    Args:
        prompt (str): The prompt to display to the user. Do not include a colon or newline at the end.
        base_path (str): The base path of the files. If None, the function will prompt the user to enter it.
        randomize (bool): Whether to randomly sample the files. If False, the user will be prompted to manually enter the file paths. Defaults to False.

    Returns:
        list[str]: The list of files in the directory.
    """
    while base_path is None:
        base_path = input("Enter the base path of the files, or leave it blank if the files are in the current directory: ")
        if base_path == "":
            base_path = os.getcwd()
        if os.path.isdir(base_path):
            break
        else:
            print("Please enter a valid directory.")
            base_path = None
    
    if randomize:
        sample_size = get_int("Enter the number of files to sample: ")
        possible_files = [file for file in os.listdir(base_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))]
        if len(possible_files) < sample_size:
            print(f"There are only {len(possible_files)} files in the directory. Returning all files.")
            selected_files = possible_files
        else:
            selected_files = random.sample(possible_files, sample_size)
        
        # Return full paths, not just filenames
        return [os.path.join(base_path, file) for file in selected_files]

    files = []
    print(prompt)
    print("Type Q to finish.")
    while True:
        file = input()
        if file == "Q":
            break
        if os.path.isfile(os.path.join(base_path, file)):
            files.append(os.path.join(base_path, file))
        else:
            print("Please enter a valid file.")
    return files


def is_image(file_path: str) -> bool:
    """
    Checks if a file is an image.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')) 