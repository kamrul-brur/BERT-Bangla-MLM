import re
import pandas as pd


def get_text_list_from_files(file_path):
    """Given a dataset path, return a list of cleaned lines of the text.

    Args:
        file_path (str): dataset path

    Returns:
        text_list (list): List of text lines with foreign words removed and line breaks replaced with spaces.
    """
    text_list = []
    file = open(file_path, encoding="utf8")
    # with open(file_path) as f:
    for line in file:
        if line != '\n':
            # remove foreign characters
            foreign_words_removed = "".join(
                i for i in line if i in [".", "।"] or 2432 <= ord(i) <= 2559 or ord(i) == 32)
            re.sub(' +', ' ', foreign_words_removed)
            # allow text with length greater than 10
            if len(foreign_words_removed) > 10:
                text_list.append(foreign_words_removed)
    return text_list


def get_data_from_text_files(text_path):
    """Get a list of cleaned texts, convert it to a pandas DataFrame, shuffle the data, and return the DataFrame.

    Args:
        text_path (str): Path to the dataset text file(s).

    Returns:
        dataframe (pd.DataFrame): DataFrame of text data.
    """

    texts = get_text_list_from_files(text_path)
    df = pd.DataFrame(
        {
            "text": texts
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df
