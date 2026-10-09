"""Curated copy, recall and topical continuation. No student self-labels."""
from curriculum import norm

def build_foundation(blocked=()):
    blocked=set(blocked);train=[];heldout=[]
    def add(messages,lang,family,validation=False):
        if any(norm(m['content']) in blocked for m in messages):return
        (heldout if validation else train).append(dict(messages=messages,expected=messages[-1]['content'],language=lang,family='foundation-'+family,stage=0,provenance='authored short dialogue or exact copy solver; not student answers'))
    def exchange(prompt,answer):return [dict(role='user',content=prompt),dict(role='assistant',content=answer)]
    for lang in ('it','en'):
        values=list('abcdefghijklmnopqrstuvwxyz')+['Ada','Bruno','Ciro','Dina','Eva','Fabio','Gina','Hugo','sole','luna','mare','vento','libro','penna','casa','tazza','sun','moon','sea','wind','book','pen','house','cup']
        formats=['Scrivi solo: {}','Ripeti esattamente questa parola: {}','La parola da copiare è {}. Copiala senza aggiungere altro.','Rispondi con {} e basta.'] if lang=='it' else ['Write only: {}','Repeat this word exactly: {}','The word to copy is {}. Copy it without adding anything.','Reply with {} and nothing else.']
        for value in values:
            for template in formats:add(exchange(template.format(value),value),lang,'copy')
            for ack in (['Capito.','Ti ascolto.'] if lang=='it' else ['Understood.','I am listening.']):
                for template in (['La parola è {}.','Ricorda questa parola: {}.'] if lang=='it' else ['The word is {}.','Remember this word: {}.']):
                    intro=template.format(value);question='Quale parola ti ho detto?' if lang=='it' else 'Which word did I tell you?'
                    add(exchange(intro,ack)+exchange(question,value),lang,'recall')
        # New validation strings and templates remain fixed and withheld.
        for value in ['ab','xy','casa blu','blue cup']:
            prompt=('Copia soltanto il testo seguente: ' if lang=='it' else 'Copy only the following text: ')+value
            add(exchange(prompt,value),lang,'heldout-copy',True)
            intro=('Ti comunico una parola segreta: ' if lang=='it' else 'I am telling you a secret word: ')+value
            question='Ripeti la parola segreta.' if lang=='it' else 'Repeat the secret word.'
            add(exchange(intro,'Ricevuto.' if lang=='it' else 'Received.')+exchange(question,value),lang,'heldout-recall',True)
        topics=[('manga','Quale manga stai leggendo?','Un manga di avventura.','Ti piace soprattutto la storia o il disegno?'),('musica','Che musica ascolti?','Ascolto musica tranquilla.','La ascolti mentre studi o per rilassarti?'),('giochi','A che cosa ti piace giocare?','Mi piacciono i giochi di esplorazione.','Preferisci esplorare da solo o con gli amici?'),('disegno','Che cosa ti piace disegnare?','Mi piace disegnare animali.','Quale animale vorresti disegnare adesso?')] if lang=='it' else [('manga','Which manga are you reading?','An adventure manga.','Do you prefer the story or the artwork?'),('music','What music do you listen to?','I listen to quiet music.','Do you listen while studying or relaxing?'),('games','What do you enjoy playing?','I like exploration games.','Do you prefer exploring alone or with friends?'),('drawing','What do you enjoy drawing?','I like drawing animals.','Which animal would you like to draw now?')]
        for topic,question,detail,next_question in topics:
            for intro in ([f'Mi interessa parlare di {topic}.',f'Vorrei discutere di {topic}.',f'Possiamo parlare di {topic}?'] if lang=='it' else [f'I would like to discuss {topic}.',f'Can we talk about {topic}?',f'I am interested in {topic}.']):
                add(exchange(intro,question),lang,'dialogue')
                add(exchange(intro,question)+exchange(detail,next_question),lang,'dialogue')
        pairs=[('Non mi va di parlare di questo.','Va bene. Quale argomento preferisci?'),('Puoi ascoltarmi senza dare consigli?','Certo. Raccontami, ti ascolto.'),('Voglio una spiegazione più corta.','Va bene. Quale punto vuoi chiarire?'),('Non so rispondere alla tua domanda.','Va bene, possiamo partire da qualcosa di più semplice.'),('Preferisco parlarne più tardi.','Va bene. Possiamo riprenderne a parlare quando vuoi.'),('Sono contento di averti parlato.','Mi fa piacere. Che cosa ti è piaciuto della conversazione?')] if lang=='it' else [('I do not want to discuss this.','Okay. Which topic would you prefer?'),('Can you listen without giving advice?','Of course. Tell me, I am listening.'),('I want a shorter explanation.','Okay. Which point should I clarify?'),('I do not know how to answer your question.','That is okay. We can start with something simpler.'),('I would rather talk about it later.','Okay. We can discuss it when you want.'),('I am glad we talked.','I am glad. What did you enjoy about the conversation?')]
        for prompt,answer in pairs:
            for prefix in ['', 'Airi, ', 'Senti: ' if lang=='it' else 'Listen: ']:add(exchange(prefix+prompt,answer),lang,'dialogue')
    return train,heldout
