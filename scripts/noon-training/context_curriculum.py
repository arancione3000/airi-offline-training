"""Expand explicit context tasks, with solver answers rather than student guesses."""
from curriculum import norm

def build_context(blocked=()):
    blocked=set(blocked);train=[];heldout=[]
    for lang in ('it','en'):
        for validation,values in ((False,['Ada','Bruno','Ciro','Dina','Eva','Fabio','Gina','Hugo']),(True,['Lidia','Nereo'])):
            for task in ('name','location','topic','correction'):
                for value in values:
                    for variant in range(2 if validation else 32):
                        if lang=='it':
                            prefix='Per questa prova: ' if validation else ['', 'Nota: ', 'Ti dico una cosa: ', 'Ricorda: '][variant%4]
                            intro={'name':f'Il mio nome è {value}.','location':f'La scatola è nella stanza {value}.','topic':f'Parliamo del progetto {value}.','correction':f'Prima ho detto Zeno, ma il nome corretto è {value}.'}[task]
                            question={'name':'Quale nome ti ho detto?','location':'In quale stanza è la scatola?','topic':'Di quale progetto parliamo?','correction':'Qual è il nome corretto?'}[task]
                            reply=['Ho capito.','Va bene.','Ti ascolto.','Ricevuto.'][variant if validation else (variant//4)%4]
                        else:
                            prefix='For this test: ' if validation else ['', 'Note: ', 'Let me tell you something: ', 'Remember: '][variant%4]
                            intro={'name':f'My name is {value}.','location':f'The box is in room {value}.','topic':f'We are discussing project {value}.','correction':f'I said Zeno earlier, but the correct name is {value}.'}[task]
                            question={'name':'Which name did I tell you?','location':'Which room contains the box?','topic':'Which project are we discussing?','correction':'What is the correct name?'}[task]
                            reply=['I understand.','Okay.','I am listening.','Understood.'][variant if validation else (variant//4)%4]
                        messages=[dict(role='user',content=prefix+intro),dict(role='assistant',content=reply)]
                        if variant>=16 and not validation:
                            messages += [dict(role='user',content='Aspetta.' if lang=='it' else 'Wait.'),dict(role='assistant',content='Va bene.' if lang=='it' else 'Okay.')]
                        messages += [dict(role='user',content=question),dict(role='assistant',content=value)]
                        if any(norm(m['content']) in blocked for m in messages):continue
                        row=dict(messages=messages,expected=value,language=lang,family='context-'+task,stage=0 if variant<4 else 1 if variant<16 else 2,pair_id=f'{lang}:{task}:{variant}',provenance='deterministic explicit context solver; not student labels')
                        (heldout if validation else train).append(row)
    return train,heldout
