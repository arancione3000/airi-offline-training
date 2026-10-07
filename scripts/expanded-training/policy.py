"""Finite larger-corpus experiment, deterministic across restarts."""
import random

START = 17203
PILOT = START + 512
TARGET = START + 32768


class DeckSampler:
    def __init__(self):
        self.cache = {}

    def index(self, language, cursor, size):
        epoch, offset = divmod(cursor, size)
        key = (language, epoch, size)
        if key not in self.cache:
            deck = list(range(size))
            random.Random(f'airi-expanded-71027-{language}-{epoch}').shuffle(deck)
            self.cache = {k:v for k,v in self.cache.items() if k[0]!=language}
            self.cache[key] = deck
        return self.cache[key][offset]


def group_for(step):
    update = step - START
    if not 0 <= update < TARGET-START:
        raise ValueError('update outside expanded recipe')
    if update % 4 == 3:
        return 'dialogue-'+('it' if (update//4)%2==0 else 'en')
    return 'corpus-'+('en' if update%4==1 else 'it')


def pilot_decision(before, after, retention, degeneration):
    language_gain = before['language']['language_nll']-after['language']['language_nll']
    corpus_ok = all(after['corpus'][lang] < before['corpus'][lang] for lang in ['it','en'])
    dev_ok = all(after['development'][lang]['nll_per_byte'] <= before['development'][lang]['nll_per_byte'] for lang in ['it','en'])
    # A larger run is conditional on measured improvement, never just elapsed time.
    return {'passed': bool(retention[0] and degeneration[0] and corpus_ok and dev_ok and language_gain>=.005),
            'retention': list(retention), 'degeneration': list(degeneration),
            'corpus_improved_both_languages': corpus_ok, 'development_nonregression': dev_ok,
            'language_nll_gain': language_gain, 'minimum_language_nll_gain': .005}
