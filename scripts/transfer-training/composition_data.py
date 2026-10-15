"""Exact solver curriculum: vary values, lengths, wording, histories and roles.

The old development sets are untouched. A new final-only set uses reserved
values AND templates. Neither set is a source of labels for training.
"""
import random
from curriculum import norm

PREVIOUS_SEALED_VALUES=('qv7m','rt8n','wk6p','jd9s','fv3r','hm4t','np5w','sx2k',
               'silver orchard','quiet pebble','amber forest','violet river',
               'gentle sunrise','winter garden','hidden meadow','distant island')
SEALED_VALUES=('zb6q','vk3x','mq9f','px5r','gt7w','bn2j','yz8d','cr4h',
               'copper window','patient willow','frozen canvas','sudden lantern',
               'velvet canyon','wooden compass','dancing mountain','silent harbor')
PROTECTED_VALUES=('ab','xy','casa blu','blue cup','Lidia','Nereo','Luca','Sara','Marco','Anna','Giulia','Paolo','Elena','Andrea','Sofia','Davide','Marta','Nina')
WORDS=('alba','neve','fiore','ponte','sasso','verde','rosso','calmo','piccolo','nuovo',
       'dawn','snow','flower','bridge','stone','green','red','calm','small','new')

def exchange(prompt,answer):
    return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]

def build_composition(blocked=()):
    blocked=set(blocked);train=[];sealed=[];rng=random.Random(20261011)
    reserved={norm(v) for v in SEALED_VALUES+PREVIOUS_SEALED_VALUES+PROTECTED_VALUES}
    copies={
      'it':('Trascrivi questo testo senza commenti: {}','Riporta esattamente: {}',
            'Testo da ripetere: {}','Scrivi soltanto il contenuto fra parentesi: ({})',
            'Il codice da restituire è {}. Non aggiungere spiegazioni.',
            'Ecco una sequenza: {}. Rispondi con la stessa sequenza.'),
      'en':('Transcribe this text without comments: {}','Return exactly: {}',
            'Text to repeat: {}','Write only the content inside parentheses: ({})',
            'The code to return is {}. Add no explanation.',
            'Here is a sequence: {}. Reply with the same sequence.')}
    recalls={
      'it':(('Il codice da ricordare è {}.','Restituisci il codice che ti ho dato.'),
            ('Memorizza il testo {}.','Quale testo dovevi memorizzare?'),
            ('La mia etichetta è {}.','Scrivi la mia etichetta.'),
            ('Il messaggio riservato contiene {}.','Che cosa contiene il messaggio riservato?')),
      'en':(('The code to remember is {}.','Return the code I gave you.'),
            ('Memorize the text {}.','Which text were you asked to memorize?'),
            ('My label is {}.','Write my label.'),
            ('The private message contains {}.','What does the private message contain?'))}
    for lang in ('it','en'):
        for i in range(3072):
            bucket=i%3
            while True:
                if bucket==0:value=''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=rng.randint(1,3)))
                elif bucket==1:value=''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789',k=rng.randint(4,12)))
                else:value=' '.join(rng.sample(WORDS,rng.randint(2,3)))
                if norm(value) not in reserved and norm(value) not in blocked:break
            for task in ('copy','recall'):
                if task=='copy':messages=exchange(copies[lang][(i//3)%len(copies[lang])].format(value),value)
                else:
                    intro,question=recalls[lang][(i//3)%len(recalls[lang])]
                    ack=('Annotato.','D’accordo.','Lo ricorderò.','Capito.') if lang=='it' else ('Noted.','All right.','I will remember.','Understood.')
                    messages=exchange(intro.format(value),ack[(i//4)%len(ack)])
                    if i%4==0:
                        messages+=exchange('Aspetta un momento.' if lang=='it' else 'Wait a moment.',
                                           'Va bene.' if lang=='it' else 'Okay.')
                    messages+=exchange(question,value)
                if i%8==0:
                    messages.insert(0,dict(role='system',content='Segui le istruzioni e rispondi brevemente.' if lang=='it' else 'Follow the instructions and answer briefly.'))
                if any(norm(m['content']) in blocked for m in messages):continue
                train.append(dict(messages=messages,expected=value,language=lang,family='foundation-'+task,stage=0,
                                  length_bucket=bucket,provenance='deterministic exact copy/recall solver v2; no model labels'))
        for value in SEALED_VALUES:
            copy=('Rendi identica la stringa delimitata da «{}».' if lang=='it' else 'Reproduce the string delimited by «{}».').format(value)
            intro=('Il contenuto del biglietto è «{}».' if lang=='it' else 'The note contains «{}».').format(value)
            question='Che cosa era scritto sul biglietto? Solo il contenuto.' if lang=='it' else 'What was written on the note? Only the content.'
            for task,messages in (('copy',exchange(copy,value)),('recall',exchange(intro,'Letto.' if lang=='it' else 'Read.')+exchange(question,value))):
                sealed.append(dict(messages=messages,expected=value,language=lang,family='sealed-'+task,provenance='final-only exact solver; never train or adapt'))
    return train,sealed

def sampling_pools(rows):
    pools={lang:{} for lang in ('it','en')}
    for row in rows:
        if row['family'] not in ('foundation-copy','foundation-recall'):continue
        value=row['expected']
        bucket=row.get('length_bucket',2 if ' ' in value else 0 if len(value)<=3 else 1)
        pools[row['language']].setdefault(f'{row["family"]}:{bucket}',[]).append(row)
    if any(len(v)!=6 or any(not rows for rows in v.values()) for v in pools.values()):
        raise ValueError('incomplete copy/recall length coverage')
    return pools
