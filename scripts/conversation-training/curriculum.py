"""Auditable authored examples. No downloaded/generated model or self-labeling."""
import hashlib
import json
import re

def norm(text):
    return ' '.join(text.casefold().split())

def build(blocked=()):
    blocked = set(blocked)
    train, validation = [], []
    def add(messages, language, expected=None, heldout=False, family='authored'):
        if any(norm(m['content']) in blocked for m in messages):
            return
        row = {'messages': messages, 'language': language, 'family': family,
               'expected': expected or messages[-1]['content'],
               'provenance': 'authored deterministic curriculum v1; not model self-generated'}
        (validation if heldout else train).append(row)
    pairs = {
      'it': [('Ciao!', 'Ciao! Come stai?'), ('Come ti chiami?', 'Mi chiamo Airi.'),
             ('Sono stanco.', 'Ti va di riposare un po\u2019?'), ('Sono triste.', 'Mi dispiace. Vuoi raccontarmi cosa è successo?'),
             ('Grazie!', 'Prego!'), ('Non ho capito.', 'Va bene, proviamo un passo alla volta.'),
             ('Cos\u2019è un gatto?', 'Un gatto è un piccolo mammifero domestico.'),
             ('Quanti giorni ha una settimana?', 'Una settimana ha sette giorni.'),
             ('Quanti mesi ha un anno?', 'Un anno ha dodici mesi.'),
             ('Di che colore è una banana matura?', 'Una banana matura è gialla.'),
             ('Che cosa fai?', 'Sto parlando con te.'), ('Buonanotte!', 'Buonanotte, dormi bene!'),
             ('Ho bisogno di aiuto.', 'Certo. Dimmi cosa ti serve.'),
             ('Spiegamelo in modo semplice.', 'Certo, userò parole semplici.'),
             ('Preferisco una risposta breve.', 'Va bene, sarò breve.'),
             ('Non lo sai?', 'Se non lo so, te lo dico. Non voglio inventare una risposta.')],
      'en': [('Hello!', 'Hello! How are you?'), ('What is your name?', 'My name is Airi.'),
             ('I am tired.', 'Would you like to rest for a while?'), ('I am sad.', 'I am sorry. Would you like to tell me what happened?'),
             ('Thank you!', 'You are welcome!'), ('I do not understand.', 'That is okay. Let us try one step at a time.'),
             ('What is a cat?', 'A cat is a small domestic mammal.'),
             ('How many days are in a week?', 'A week has seven days.'),
             ('How many months are in a year?', 'A year has twelve months.'),
             ('What color is a ripe banana?', 'A ripe banana is yellow.'),
             ('What are you doing?', 'I am talking with you.'), ('Good night!', 'Good night, sleep well!'),
             ('I need help.', 'Of course. Tell me what you need.'),
             ('Explain it simply.', 'Of course, I will use simple words.'),
             ('I prefer a short answer.', 'Okay, I will be brief.'),
             ('Do you not know?', 'If I do not know, I will say so. I do not want to invent an answer.')]
    }
    for lang, values in pairs.items():
        prefixes = ['', 'Airi, ', 'Ehi, ' if lang=='it' else 'Hey, ', 'Per favore: ' if lang=='it' else 'Please: ']
        for index, (prompt, answer) in enumerate(values):
            for prefix in prefixes:
                add([{'role':'user','content':prefix+prompt},{'role':'assistant','content':answer}],lang,family=f'phrase-{lang}-{index}')
            prefix = 'Vorrei chiederti: ' if lang=='it' else 'I would like to ask: '
            add([{'role':'user','content':prefix+prompt},{'role':'assistant','content':answer}],lang,heldout=True,family=f'phrase-{lang}-{index}')
        for a in range(1,25):
            for b in range(1,13):
                heldout = a>20
                prompt = f'Quanto fa {a} + {b}?' if lang=='it' else f'What is {a} + {b}?'
                add([{'role':'user','content':prompt},{'role':'assistant','content':str(a+b)}],lang,heldout=heldout,family='addition-unseen-operands' if heldout else 'addition')
        names = ['Luca','Sara','Marco','Anna','Giulia','Paolo','Elena','Andrea','Sofia','Davide','Marta','Nina']
        for i,name in enumerate(names):
            intro = f'Mi chiamo {name}.' if lang=='it' else f'My name is {name}.'
            reply = f'Ciao {name}!' if lang=='it' else f'Hello {name}!'
            question = 'Come mi chiamo?' if lang=='it' else 'What is my name?'
            add([{'role':'user','content':intro},{'role':'assistant','content':reply},
                 {'role':'user','content':question},{'role':'assistant','content':name}],lang,heldout=i>=10,family='memory-unseen-name' if i>=10 else 'memory')
    # Whole normalized conversations and user prompts must be disjoint.
    key = lambda r: json.dumps(r['messages'],ensure_ascii=False,sort_keys=True)
    assert not {key(r) for r in train}&{key(r) for r in validation}
    train_prompts = {norm(r['messages'][0]['content']) for r in train}
    assert not train_prompts&{norm(r['messages'][0]['content']) for r in validation}
    return train, validation

def digest(rows):
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def exact_correct(response, expected):
    # Exact answer, ignoring only case, whitespace, and terminal punctuation.
    return norm(response).rstrip('.!?') == norm(expected).rstrip('.!?')

def repetitive(response):
    words = re.findall(r'\w+',response.casefold())
    grams = [tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    return 1-len(set(grams))/len(grams) if grams else 0.

def pilot_eligible(before, after):
    return (after['exact_correct'] >= before['exact_correct']+2
            and all(after['nll'][lang] < before['nll'][lang] for lang in ['it','en'])
            and after['repetition'] <= max(.20,before['repetition']))

def choose_winner(records):
    eligible = [r for r in records if r['eligible']]
    return max(eligible,key=lambda r:(r['after']['exact_correct'],-sum(r['after']['nll'].values()))) if eligible else None
