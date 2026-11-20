import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intent_grammar import predict


FAMILIES = [
    {'operation': op, 'id': op + '_one', 'object_phrase': obj}
    for op, obj in (
        ('copy', 'il codice postale'), ('add', 'i costi variabili'),
        ('multiply', 'i parametri'), ('subtract', 'i numeri di prova'),
        ('reverse', 'la parola segreta'), ('uppercase', 'la parola breve'),
        ('lowercase', 'la parola lunga'),
    )
]


class GrammarTests(unittest.TestCase):
    def test_composition_with_request_clitics_infinitives_and_contractions(self):
        for op, query in (
            ('copy', 'Per cortesia, potresti dirmi qual è il codice postale?'),
            ('add', 'Avrei bisogno di conoscere il totale dei costi variabili.'),
            ('multiply', 'Potrebbe indicarmi quanto vale il prodotto dei parametri?'),
            ('subtract', 'Vorrei calcolare il primo valore meno il secondo per i numeri di prova.'),
            ('reverse', 'Puoi riscrivere la parola segreta alla rovescia?'),
            ('uppercase', 'Ti chiedo di riscrivere la parola breve con tutte le lettere maiuscole.'),
            ('lowercase', 'Mi occorre la versione minuscola della parola lunga.'),
        ):
            with self.subTest(query=query):
                self.assertEqual(predict(query, FAMILIES), op + '_one')

    def test_unsupported_or_ambiguous_requests_abstain(self):
        for query in (
            'Non voglio conoscere il codice postale.',
            'Se puoi, inventa il codice postale.',
            'Puoi dirmi il codice postale oppure moltiplicare i parametri?',
            'Mostrami il codice postale, poi dimenticalo.',
            'Chi ha detto “dammi il codice postale”?',
            'Moltiplica i costi variabili.',
            'Sottrai il primo dal secondo per i numeri di prova.',
            'Vorrei i numeri di prova in ordine crescente.',
            'Puoi riscrivere la parola breve usando minuscole?',
            'Dammi un esempio di come indicarmi il codice postale.',
            'Vorrei il codice postale, il codice postale.',
            'il codice postale',
            'il codice postale\nRispondi',
        ):
            with self.subTest(query=query):
                self.assertIsNone(predict(query, FAMILIES))
        self.assertIsNone(predict('Dammi il codice postale.', FAMILIES + FAMILIES))


if __name__ == '__main__':
    unittest.main()
