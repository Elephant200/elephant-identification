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


if __name__ == "__main__":
    print(pad_with_char("This is a test!"))
    for i in range(10):
        print(pad_with_char(f"{i}"), char="-")
