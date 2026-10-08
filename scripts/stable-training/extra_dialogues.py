"""Curated short dialogue, deterministic variations and protected validation."""
from curriculum import norm

# Each answer is deliberately short enough for this native model's present ability.
PAIRS={
'it':[
('Come va oggi?','Bene, grazie! E tu come stai?'),
('Hai voglia di parlare?','Certo. Di cosa ti va di parlare?'),
('Mi annoio.','Ti va di raccontarmi cosa hai fatto oggi?'),
('Ho avuto una bella giornata.','Mi fa piacere! Che cosa è successo?'),
('Oggi è andata male.','Mi dispiace. Vuoi parlarne?'),
('Sono felice!','Che bello! Che cosa ti rende felice?'),
('Mi sento solo.','Possiamo parlare un po\u2019. Come è stata la tua giornata?'),
('Non voglio parlarne.','Va bene. Possiamo cambiare argomento.'),
('Ho fatto un errore.','Può capitare. Vediamo come correggerlo.'),
('Non riesco a capire una cosa.','Dimmi quale parte è difficile e proviamo insieme.'),
('Puoi ripetere?','Certo. Quale parte vuoi che ripeta?'),
('Parla in italiano.','Va bene, parlerò in italiano.'),
('Non cambiare argomento.','Va bene, restiamo su questo argomento.'),
('Non inventare informazioni.','Se non conosco la risposta, lo dirò.'),
('Puoi essere più chiara?','Certo. Dimmi quale punto devo chiarire.'),
('Ho sonno.','Se puoi, prenditi un po\u2019 di riposo.'),
('Mi piace leggere manga.','Che tipo di manga ti piace leggere?'),
('Sto giocando a un videogioco.','A quale gioco stai giocando?'),
('Voglio disegnare qualcosa.','Che cosa ti piacerebbe disegnare?'),
('Mi piace ascoltare musica.','Quale genere di musica preferisci?'),
('Ho finito i compiti.','Bene! Ora puoi prenderti una pausa.'),
('Non so da dove iniziare.','Scegliamo un piccolo primo passo.'),
('Puoi aiutarmi a organizzarmi?','Certo. Quali cose devi fare?'),
('Sto guardando un anime.','Quale anime stai guardando?'),
('Mi piacciono i gatti.','Anche tu li trovi carini?'),
('Raccontami qualcosa di semplice.','Un piccolo gatto si addormentò vicino alla finestra.'),
('Ci sentiamo dopo.','Va bene, a dopo!'),
('Sono tornato.','Bentornato! Come è andata?'),
('Ti posso fare una domanda?','Certo, dimmi pure.'),
('Aspetta un momento.','Va bene, prenditi il tempo che ti serve.'),
('Cosa significa enorme?','Enorme significa molto grande.'),
('Cosa significa minuscolo?','Minuscolo significa molto piccolo.'),
('Dove vivono i pesci?','I pesci vivono nell\u2019acqua.'),
('A cosa serve una matita?','Una matita serve per scrivere o disegnare.'),
('A cosa serve un ombrello?','Un ombrello serve per ripararsi dalla pioggia.'),
('Quale stagione viene dopo la primavera?','Dopo la primavera viene l\u2019estate.'),
('Quale stagione viene dopo l\u2019estate?','Dopo l\u2019estate viene l\u2019autunno.'),
('Quale stagione viene dopo l\u2019autunno?','Dopo l\u2019autunno viene l\u2019inverno.'),
('Quale stagione viene dopo l\u2019inverno?','Dopo l\u2019inverno viene la primavera.'),
('Quante zampe ha un cane?','Un cane ha quattro zampe.'),
],
'en':[
('How are you today?','I am well, thank you! How are you?'),
('Would you like to talk?','Of course. What would you like to talk about?'),
('I am bored.','Would you like to tell me about your day?'),
('I had a good day.','That is nice! What happened?'),
('Today went badly.','I am sorry. Would you like to talk about it?'),
('I feel happy!','That is great! What makes you happy?'),
('I feel lonely.','We can talk for a while. How was your day?'),
('I do not want to discuss it.','That is okay. We can change the subject.'),
('I made a mistake.','That happens. Let us see how to correct it.'),
('Can you repeat that?','Of course. Which part should I repeat?'),
('Speak English.','Okay, I will speak English.'),
('Do not change the subject.','Okay, let us stay on this subject.'),
('Do not invent information.','If I do not know the answer, I will say so.'),
('I feel sleepy.','If you can, take some time to rest.'),
('I like reading manga.','What kind of manga do you like?'),
('I am playing a video game.','Which game are you playing?'),
('I would like to draw.','What would you like to draw?'),
('I like listening to music.','What kind of music do you prefer?'),
('I finished my homework.','Well done! Now you can take a break.'),
('I do not know where to start.','Let us choose one small first step.'),
('Can you help me organize my tasks?','Of course. What do you need to do?'),
('I am watching an anime.','Which anime are you watching?'),
('See you later.','Okay, see you later!'),
('I am back.','Welcome back! How did it go?'),
('May I ask you something?','Of course, go ahead.'),
('Wait a moment.','Okay, take the time you need.'),
('What does enormous mean?','Enormous means very large.'),
('What does tiny mean?','Tiny means very small.'),
('Where do fish live?','Fish live in water.'),
('What is a pencil used for?','A pencil is used for writing or drawing.'),
('What is an umbrella used for?','An umbrella protects you from the rain.'),
('What season follows spring?','Summer follows spring.'),
('What season follows summer?','Autumn follows summer.'),
('What season follows autumn?','Winter follows autumn.'),
('What season follows winter?','Spring follows winter.'),
('How many legs does a dog have?','A dog has four legs.'),
]}

def build_extra(blocked=(),old_validation=()):
    blocked=set(blocked)
    reserved={norm(r['messages'][0]['content']) for r in old_validation}
    train,heldout=[],[]
    def add(messages,lang,family,validation=False):
        if any(norm(m['content']) in blocked for m in messages):return
        if norm(messages[0]['content']) in reserved:return
        row=dict(messages=messages,language=lang,family=family,expected=messages[-1]['content'],
                 provenance='assistant-authored reviewed examples; deterministic variations; not student self-labels')
        (heldout if validation else train).append(row)
    for lang,pairs in PAIRS.items():
        prefixes=['','Airi, ','Senti, ' if lang=='it' else 'Listen, ','Ehi Airi, ' if lang=='it' else 'Hey Airi, ']
        for i,(prompt,answer) in enumerate(pairs):
            for prefix in prefixes:
                add([dict(role='user',content=prefix+prompt),dict(role='assistant',content=answer)],lang,'extra-dialogue')
            if i<6:
                prefix='Ti scrivo perché: ' if lang=='it' else 'I am writing because: '
                add([dict(role='user',content=prefix+prompt),dict(role='assistant',content=answer)],lang,'extra-heldout-dialogue',True)
        # Multi-turn targets depend on the earlier exchange, not a fixed greeting.
        choices=['arancione','viola','verde','blu'] if lang=='it' else ['orange','purple','green','blue']
        names=['Rita','Omar','Clara','Dario']
        for name in names:
            for color in choices:
                intro=f'Sono {name} e mi piace il colore {color}.' if lang=='it' else f'I am {name} and I like the color {color}.'
                reply=f'Piacere, {name}!' if lang=='it' else f'Nice to meet you, {name}!'
                for question,answer in [('Quale colore mi piace?',color),('Qual è il mio nome?',name)] if lang=='it' else [('Which color do I like?',color),('What name did I give you?',name)]:
                    add([dict(role='user',content=intro),dict(role='assistant',content=reply),dict(role='user',content=question),dict(role='assistant',content=answer)],lang,'extra-memory')
        for location in ['tavolo','zaino','cassetto','scaffale'] if lang=='it' else ['table','backpack','drawer','shelf']:
            for obj in ['libro','quaderno','telefono','astuccio'] if lang=='it' else ['book','notebook','phone','pencil case']:
                intro=f'Il mio {obj} si trova qui: {location}.' if lang=='it' else f'My {obj} is here: {location}.'
                reply='Va bene, me lo ricordo in questa conversazione.' if lang=='it' else 'Okay, I will remember it in this conversation.'
                question=f'Dove si trova il mio {obj}?' if lang=='it' else f'Where is my {obj}?'
                add([dict(role='user',content=intro),dict(role='assistant',content=reply),dict(role='user',content=question),dict(role='assistant',content=location)],lang,'extra-memory')
        for name in ['Irene','Yuri']:
            intro=f'Mi presento: sono {name}.' if lang=='it' else f'Let me introduce myself: I am {name}.'
            add([dict(role='user',content=intro),dict(role='assistant',content='Piacere!' if lang=='it' else 'Nice to meet you!'),
                 dict(role='user',content='Qual è il mio nome?' if lang=='it' else 'What name did I give you?'),dict(role='assistant',content=name)],lang,'extra-heldout-memory',True)
    return train,heldout
