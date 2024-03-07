import unittest
from approvaltests import verify_as_json
from src.snippet_generation import split_into_snippets


class SnippetGeneratorIOTest(unittest.TestCase):
    test_documents = [
        "The Hubble telescope discovered two moons of Pluto, Nix and Hydra.",
        "Edwin Hubble, an astronomer with great achievement, completely reimagined our place in the universe (the "
        "telescope is named by him).",
        "Press ESC key, then the : (colon), and type the wq command after the colon and hit the Enter key to save and "
        "leave Vim.",
        "In Vim, you can always press the ESC key on your keyboard to enter the normal mode in the Vim editor.",
        "Common heart attack symptoms include: (1) Chest pain, (2) Pain or discomfort that spreads to the shoulder, "
        "arm, back, neck, jaw, teeth or sometimes the upper belly, etc.",
        "A heart attack happens when the flow of blood that brings oxygen to your heart muscle suddenly becomes "
        "blocked."
    ]

    def test_split_into_snippets_output(self):
        actual = [split_into_snippets(document) for document in self.test_documents]
        verify_as_json(actual)


if __name__ == '__main__':
    pass
    #unittest.main()
