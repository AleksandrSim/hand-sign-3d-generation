import pytest

from src.control.language_engine import transform_word, transform_to_list


@pytest.mark.parametrize(
    'input_string, expected_result',
    [('Hello', ['H', 'E', 'L', 'L', 'O']),
     ('', [])],
)
def test_transform_word(input_string: str, expected_result: list[str]):
    result = transform_word(input_string)
    assert result == expected_result


@pytest.mark.parametrize(
    'input_string, special_sequences, expected_result',
    [('Aleks', [], ['A', 'L', 'E', 'K', 'S']),
     ('<name> Aleks', ['name', 'from'],
      ['name', 'prob', 'A', 'L', 'E', 'K', 'S']),
     ('Aleks  I', [], ['A', 'L', 'E', 'K', 'S', 'prob', 'I']),
     ('<name> <from>', ['name', 'from'], ['name', 'prob', 'from'])]
)
def test_transform_to_list(input_string: str, special_sequences: list[str],
                           expected_result: list[str]):
    assert transform_to_list(
        input_string, special_sequences) == expected_result
