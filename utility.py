import os

def pad_with_char(*values: str, char: str = "=", sep: str = " ") -> str:
    """
    Pads a string with a character to the left and right to center it in the terminal.

    Args:
        *values (str): The strings to pad. If multiple strings are provided, they are joined with the separator.
        char (str): The character to pad with. Defaults to "=".
        sep (str): The separator between the strings. Defaults to " ".

    Returns:
        str: The padded string.
    """
    text = sep.join(values)
    terminal_width = os.get_terminal_size().columns
    padding = (terminal_width - len(text)) // 2
    return f"{char * padding}{text}{char * padding}"

def print_with_padding(*values: str, char: str = "=", sep: str = " ") -> None:
    """
    Prints a string with a character to the left and right to center it in the terminal.

    Args:
        *values (str): The strings to print. If multiple strings are provided, they are joined with the separator.
        char (str): The character to pad with. Defaults to "=".
        sep (str): The separator between the strings. Defaults to " ".
    """
    print(pad_with_char(*values, char=char, sep=sep))

def get_int(prompt: str) -> int:
    """
    Gets an integer from the user.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        int: The integer entered by the user.
    """
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

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
            files = input(prompt)
            print("Please drag and drop files from finder into the terminal, so that they are enclosed in single quotes. Do not separate with spaces.")
            try:
                files = files[1:-1].split("''")
                if all(os.path.isfile(file) for file in files):
                    return files
                else:
                    raise FileNotFoundError("One or more files not found.")
            except:
                print("Please enter a valid list of files.")
    elif input_method == "2":
        while True:
            base_path = input("Enter the base path of the files, or leave it blank if the files are in the current directory: ")
            if base_path == "":
                base_path = os.getcwd()
            if os.path.isdir(base_path):
                break
            else:
                print("Please enter a valid directory.")
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

def get_multiple_choice(prompt: str, choices: list[str] = ["Yes", "No"], auto_lower: bool = True, first_letter_only: bool | None = None) -> str:
    """
    Gets a response from the user from a list of choices.

    Args:
        prompt (str): The prompt to display to the user.
        choices (list[str]): The list of choices to choose from. Defaults to ["Yes", "No"].
        auto_lower (bool): Whether to automatically convert the response to lowercase.
        first_letter_only (bool): Whether to work with only the first letter of the response. If None, the function will determine automatically.

    Returns:
        str: The response entered by the user.
    """
    first_letters = [choice[0].lower() if auto_lower else choice[0] for choice in choices]
    if first_letter_only is None:
        if len(set(first_letters)) != len(first_letters):
            first_letter_only = False
        else:
            first_letter_only = True
    
    modified_choices = [choice.lower() if auto_lower else choice for choice in choices]

    if first_letter_only:
        modified_choices = [choice[0] for choice in modified_choices]
    
    while True:
        choice = input(prompt).strip()
        choice = choice.lower() if auto_lower else choice
        choice = choice[0] if first_letter_only and len(choice) > 0 else choice
        if choice in modified_choices:
            return choices[modified_choices.index(choice)]
        else:
            print(f"Please enter a valid choice. {'You may ignore case. ' if auto_lower else ''}{'You may use the first letter of the choice. ' if first_letter_only else ''}Valid choices are: {', '.join(choices)}")

def get_list_of_ints(prompt: str, *, input_separator: str = ", ") -> list[int]:
    """
    Gets a list of integers from the user.

    Args:
        prompt (str): The prompt to display to the user. Include a colon or newline at the end.
        input_separator (str): The separator between the integers. Defaults to ", ".

    Returns:
        list[int]: The list of integers entered by the user.
    """
    while True:
        try:
            return list(map(int, input(prompt).split(input_separator)))
        except ValueError:
            print(f"Please enter a valid list of integers separated by \"{input_separator}\".")

if __name__ == "__main__":
    print(pad_with_char("This is a test!"))
    for i in range(10):
        print(pad_with_char(f"{i}", char="-"))
    print(get_list_of_files("Enter the path of the files"))
