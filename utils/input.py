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


def get_multiple_choice(prompt: str, choices: list[str] = ["Yes", "No"], auto_lower: bool = True, first_letter_only: bool | None = None, default_choice: str | None = None) -> str:
    """
    Gets a response from the user from a list of choices.

    Args:
        prompt (str): The prompt to display to the user.
        choices (list[str]): The list of choices to choose from. Defaults to ["Yes", "No"].
        auto_lower (bool): Whether to automatically convert the response to lowercase.
        first_letter_only (bool): Whether to work with only the first letter of the response. If None, the function will determine automatically.
        default_choice (str): The default choice to return if the user enters nothing. If None, the function will not accept a blank response.

    Returns:
        str: The response entered by the user.
    """
    first_letters = [choice[0].lower() if auto_lower else choice[0] for choice in choices]
    if len(set(first_letters)) != len(first_letters):
        first_letter_only = False
    elif first_letter_only is None:
        first_letter_only = True
    
    modified_choices = [choice.lower() if auto_lower else choice for choice in choices]

    if first_letter_only:
        modified_choices = [choice[0] for choice in modified_choices]
    
    if default_choice is not None:
        if default_choice not in choices:
            raise ValueError("Default choice must be one of the choices.")
    
    while True:
        choice = input(prompt).strip()
        choice = choice.lower() if auto_lower else choice
        choice = choice[0] if first_letter_only and len(choice) > 0 else choice
        if choice in modified_choices:
            return choices[modified_choices.index(choice)]
        elif default_choice is not None and choice == "":
            return default_choice
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