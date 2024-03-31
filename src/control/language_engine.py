"""
Tools to transform input string into sequence of characters, including
special characters, specified in angle brackets, like '<name>' or '<from>'.

The current implementation ignores all punctuation symbols and numbers.

"""


import warnings


def transform_word(word):
    return list(word.upper())


def transform_to_list(input_string: str, special_sequences: list[str])\
        -> list[str]:
    output_list = []
    word = ""
    in_angle_brackets = False

    for char in input_string:
        if char == '<':
            if word:  # Handle word before angle brackets
                output_list.extend(transform_word(word))
                word = ""
            in_angle_brackets = True
        elif char == '>':
            in_angle_brackets = False
            if word in special_sequences:
                output_list.append(word)
            else:
                warnings.warn(f'Unknown keyword: {word}')
            word = ""
        elif char.isalpha():
            word += char
        elif char.isspace():
            if word:
                output_list.extend(transform_word(word))
                word = ""
            # Avoid multiple whitespaces
            print(output_list)
            if len(output_list) > 0 and output_list[-1] != 'prob':
                output_list.append('prob')

    if word:
        print(word, in_angle_brackets)
        if in_angle_brackets and word in special_sequences:
            output_list.append(word)
        else:
            output_list.extend(transform_word(word))

    return output_list
