import os

def dir_path(path):
    """
    Check if a path is a directory.

    Parameters:
    - path (str): The path to check.

    Returns:
    - str: The validated path if it is a directory.

    Raises:
        NotADirectoryError: If the path is not a directory.
    """
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)


def tar_path(path):
    """
    Check if a path is a directory. If it is not, create the directory and return the path.

    Parameters:
    - path (str): The path to check.

    Returns:
    - str: The validated path as a directory.

    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"Output folder {path} created")
        return path
    return path