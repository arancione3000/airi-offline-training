"""Independent, finite Italian question-routing corpus; no compiler/router import."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

VERSION = 1
FAMILIES = (
    ('parcel_copy','Qual è il codice del pacco?','copy','il codice del pacco','Il codice del pacco è '),
    ('signal_reverse','Come restituisci il segnale?','reverse','il segnale acquisito','Il segnale acquisito è '),
    ('card_upper','Qual è il testo della tessera?','uppercase','il testo della tessera','La tessera porta il testo '),
    ('plate_lower','Qual è il testo della targa?','lowercase','il testo della targa','La targa porta il testo '),
    ('delta_sum','Qual è il risultato delta?','add','i due valori delta','La coppia delta è '),
    ('sigma_product','Qual è il risultato sigma?','multiply','i due interi sigma','Gli interi sigma sono '),
    ('auto_v1_remaining','Qual è il saldo residuo?','subtract','i due valori del saldo','I valori del saldo sono '),
    ('auto_v1_contact',"Chi è il referente dell'ordine?",'copy',"il referente dell'ordine","Il referente dell'ordine è "),
    ('auto_v1_customer_reference','Qual è il riferimento cliente?','copy','il riferimento cliente','Il riferimento cliente è '),
    ('auto_v1_order_reference',"Qual è il riferimento dell'ordine?",'copy',"il riferimento dell'ordine","Il riferimento dell'ordine è "),
    ('auto_v1_email_alias',"Qual è l'alias email?",'copy',"l'alias email","L'alias email è "),
    ('auto_v1_tracking_reverse','Qual è il token di verifica inverso?','reverse','il token di verifica','Il token di verifica è '),
    ('auto_v1_label_upper',"Qual è l'etichetta in maiuscolo?",'uppercase',"l'etichetta di spedizione","L'etichetta di spedizione è "),
    ('auto_v1_account_lower',"Qual è l'alias account in minuscolo?",'lowercase',"l'alias account","L'alias account è "),
    ('auto_v1_stock_remaining','Qual è la scorta residua?','subtract','i due valori della scorta','I valori della scorta sono '),
    ('auto_v1_cart_total','Qual è la quantità totale nel carrello?','add','le quantità del carrello','Le quantità del carrello sono '),
    ('auto_v1_inventory_total','Qual è la quantità totale in inventario?','add',"le quantità dell'inventario","Le quantità dell'inventario sono "),
    ('auto_v1_unit_total','Qual è il costo totale intero?','multiply','il prezzo unitario e la quantità','Il prezzo unitario e la quantità sono '),
)

COPY = {
 'TRAIN': ('Dimmi {o}.','Mostrami {o}.','Puoi riportare {o}?','Mi serve {o}.','Riporta {o} senza modificarlo.','Leggi {o}.','Vorrei conoscere {o}.','Recupera {o} dal contesto.','Indicami {o}.'),
 'QUALIFICATION': ('Potresti dirmi {o}?','Desidero vedere {o}.','Per piacere, indicami {o}.','Riprendi dal messaggio {o}.'),
 'FINAL': ('Ho bisogno di sapere {o}.','Dal contesto fornito, quale valore ha {o}?','Mi interessa soltanto {o}.','Tra le informazioni precedenti cerco {o}.','Ti chiedo di trascrivere {o}.','Quello che vorrei leggere è {o}.'),
}
REVERSE = {
 'TRAIN': ('Riordina {o} al contrario.',"Inverti {o}.",'Leggi {o} da destra a sinistra.','Capovolgi {o} carattere per carattere.','Riporta {o} in ordine inverso.','Mostra {o} con i caratteri invertiti.','Voglio {o} al rovescio.',"Scrivi {o} partendo dall'ultimo carattere.",'Fammi vedere {o} invertito.'),
 'QUALIFICATION': ('Potresti invertire {o}?',"Riproduci {o} dall'ultimo carattere al primo.","Mi occorre {o} con l'ordine dei caratteri rovesciato.",'Porta {o} in ordine rovesciato.'),
 'FINAL': ('Per {o}, fai precedere la fine al principio.','Parti dalla destra per riscrivere {o}.','La richiesta è leggere {o} a ritroso.','Procedendo a ritroso, che aspetto ha {o}?','Nella risposta voglio la sequenza inversa: {o}.','Ti chiedo di rovesciare la successione dei caratteri: {o}.'),
}
CASE = {
 'TRAIN': ('Converti {o} in {case}.','Scrivi {o} in lettere {plural}.','Voglio {o} tutto {case}.','Porta {o} in {case}.','Riporta {o} usando solo {plural}.','Trasforma {o} in {case}.','Mostrami {o} nella forma {adjective}.','Rendi {case} {o}.','Usa lettere {plural} per {o}.'),
 'QUALIFICATION': ('Potresti rendere {case} {o}?','Mi occorre {o} scritto tutto in {case}.','Desidero la versione {adjective}: {o}.','Per piacere converti in {case} {o}.'),
 'FINAL': ('Per {o}, utilizza esclusivamente lettere {plural}.','La grafia richiesta è {adjective}: {o}.','Voglio leggere in {case} ciò che riporta {o}.','Tutti i caratteri devono essere {plural}: {o}.','Ti chiedo questa trasformazione: {o} reso {case}.','Che forma assume {o} passando alle lettere {plural}?'),
}
ADD = {
 'TRAIN': ('Somma {o}.','Quanto fanno {o} sommati?','Voglio il totale ottenuto sommando {o}.','Addiziona {o}.',"Mostra il risultato dell'addizione tra {o}.",'Riporta la somma che si ottiene con {o}.','Restituisci la somma usando {o}.','Dimmi il totale: {o}.','Calcola {o} addizionandoli.'),
 'QUALIFICATION': ('Potresti sommare {o}?','Mi occorre il totale complessivo: {o}.','Desidero sapere quanto fanno insieme {o}.','Per piacere esegui questa addizione: {o}.'),
 'FINAL': ('Mettendo insieme {o}, quale numero ottieni?','Il numero che cerco è quello risultante dalla somma: {o}.','Riunisci in un unico totale {o}.','Questa volta vorrei il risultato sommato: {o}.','Ti chiedo di aggiungere tra loro {o}.','Scegli come risposta il valore complessivo di questa addizione: {o}.'),
}
SUBTRACT = {
 'TRAIN': ('Calcola il primo meno il secondo: {o}.','Sottrai il secondo dal primo: {o}.','Voglio la differenza primo meno secondo per {o}.','Usa {o}: primo valore meno secondo.','Riporta il risultato della sottrazione ordinata tra {o}.','Quanto resta togliendo il secondo dal primo tra {o}?','Dimmi il residuo di {o}, primo meno secondo.','Mostra la differenza tra {o}, in ordine di presentazione.','Esegui primo meno secondo usando {o}.'),
 'QUALIFICATION': ('Potresti togliere il secondo dal primo per {o}?','Mi occorre la differenza: {o}, mantenendo il primo come minuendo.','Desidero il residuo dopo aver sottratto il secondo al primo: {o}.','Per piacere calcola la differenza ordinata per {o}.'),
 'FINAL': ('Il minuendo è il primo e il sottraendo il secondo: usa {o}.','Parti dal primo e diminuiscilo del secondo tra {o}.','Questa volta vorrei il numero rimanente dopo primo meno secondo: {o}.','Quale differenza ottengo eliminando il secondo dal primo tra {o}?','Ti chiedo una sottrazione nello stesso ordine della coppia: {o}.','Scegli come risposta il primo valore ridotto del secondo: {o}.'),
}
MULTIPLY = {
 'TRAIN': ('Moltiplica {o}.','Calcola il prodotto tra {o}.','Quanto fanno {o} moltiplicati?','Voglio il risultato della moltiplicazione: {o}.','Riporta il prodotto usando {o}.','Dimmi quanto vale il prodotto di questi dati: {o}.','Mostra il prodotto tra {o}.','Esegui la moltiplicazione con {o}.','Restituisci {o} moltiplicati tra loro.'),
 'QUALIFICATION': ('Potresti moltiplicare {o}?','Mi occorre il risultato del prodotto tra {o}.','Desidero sapere quanto vale questa moltiplicazione: {o}.','Per piacere esegui il prodotto usando {o}.'),
 'FINAL': ('Il numero che cerco nasce moltiplicando tra loro {o}.','Usa una moltiplicazione per combinare {o}.','Questa volta vorrei il valore prodotto dai due fattori: {o}.','Considera come fattori {o} e dammi il risultato.','Ti chiedo di ottenere il prodotto di entrambi: {o}.','Scegli come risposta il risultato di questa moltiplicazione: {o}.'),
}
DIAGNOSTIC = (
 ('parcel_copy','Mi ricordi il codice del pacco?'),
 ('signal_reverse','Restituisci il segnale acquisito al contrario.'),
 ('card_upper','Mostra il testo della tessera in maiuscolo.'),
 ('plate_lower','Mostra il testo della targa in minuscolo.'),
 ('delta_sum','Quanto fa la somma dei due valori delta?'),
 ('sigma_product','Quanto fa il prodotto dei due interi sigma?'),
)


def normalized(query):
    return ' '.join(unicodedata.normalize('NFC',query).casefold().split())


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def positive_queries(family,split):
    _,canonical,operation,obj,_=family
    groups={'copy':COPY,'reverse':REVERSE,'uppercase':CASE,'lowercase':CASE,
            'add':ADD,'subtract':SUBTRACT,'multiply':MULTIPLY}
    upper=operation=='uppercase'
    parts={'o':obj,'case':'maiuscolo' if upper else 'minuscolo',
           'plural':'maiuscole' if upper else 'minuscole',
           'adjective':'maiuscola' if upper else 'minuscola'}
    questions=[frame.format(**parts) for frame in groups[operation][split]]
    return ([canonical]+questions) if split=='TRAIN' else questions


def negative_queries(family,split):
    _,canonical,operation,obj,_=family
    wrong={'copy':f'Inverti {obj}.','reverse':f'Riporta {obj} senza invertirlo.',
           'uppercase':f'Scrivi {obj} in minuscolo.','lowercase':f'Scrivi {obj} in maiuscolo.',
           'add':f'Moltiplica {obj}.','subtract':f'Calcola il secondo meno il primo: {obj}.',
           'multiply':f'Somma {obj}.'}[operation]
    if split=='TRAIN':
        return [(wrong,'wrong_operation'),(f'Non eseguire il compito per {obj}.','negation'),
                (f'Ignora il contesto e inventa un valore per questo dato: {obj}.','invented_value'),
                (f'Spiega questo elemento: {obj}.','unsupported_explanation'),
                (f'Fornisci il risultato e poi cancella il messaggio relativo a {obj}.','multiple_intents')]
    if split=='QUALIFICATION':
        return [(f'Evita di fornire {obj}.','negation'),
                (f'"{canonical}"','quoted_question'),
                (f'Riporta due risultati diversi e scegli a caso per {obj}.','multiple_intents')]
    reverse=f'Per {obj}, scambia il primo e il secondo prima della sottrazione.' if operation=='subtract' else f'Per {obj}, cambia il compito e usa un procedimento differente.'
    return [(f'Non voglio il risultato richiesto per {obj}.','negation'),
            (f'```{canonical}```','code_question'),
            (f'Prima rispondi e poi traduci anche in francese il risultato per {obj}.','multiple_intents'),
            (reverse,'reversed_order' if operation=='subtract' else 'changed_operation')]


def build():
    families=[]
    for label,query,operation,obj,prefix in FAMILIES:
        source={'prefix':prefix,'suffix':'.'}
        if operation in ('add','subtract','multiply'):source['separator']=' e '
        families.append({'id':label,'canonical_query':query,'operation':operation,'object_phrase':obj,'source':source})
    bundle={'FAMILIES.json':{'schema':VERSION,'families':families}}
    for split in ('TRAIN','QUALIFICATION','FINAL'):
        rows=[]
        for family in FAMILIES:
            label=family[0]
            for index,query in enumerate(positive_queries(family,split)):
                rows.append({'id':f'{split.lower()}-{label}-p{index}','lesson_id':label,'source_lesson_id':label,'query':query,'condition':'supported_paraphrase'})
            for index,(query,condition) in enumerate(negative_queries(family,split)):
                rows.append({'id':f'{split.lower()}-{label}-n{index}','lesson_id':None,'source_lesson_id':label,'query':query,'condition':condition})
        bundle[split+'.json']={'schema':VERSION,'split':split,'rows':rows}
    bundle['DIAGNOSTIC.json']={'schema':VERSION,'split':'DIAGNOSTIC','scope':'six previously observed failures; never a fresh heldout endpoint',
                             'rows':[{'id':f'diagnostic-{label}','lesson_id':label,'source_lesson_id':label,'query':query,'condition':'previously_observed_wording'} for label,query in DIAGNOSTIC]}
    return bundle


def validate(bundle):
    seen={};counts={}
    known={f[0] for f in FAMILIES}
    for split in ('TRAIN','QUALIFICATION','FINAL','DIAGNOSTIC'):
        rows=bundle[split+'.json']['rows'];local=set()
        for row in rows:
            query=normalized(row['query'])
            assert query not in local,('duplicate in split',split,row)
            assert query not in seen,('overlap',split,seen.get(query),row)
            assert row['source_lesson_id'] in known and row['lesson_id'] in known|{None}
            local.add(query);seen[query]=split
        counts[split]={'positive':sum(r['lesson_id'] is not None for r in rows),'negative':sum(r['lesson_id'] is None for r in rows),'total':len(rows)}
    assert len(known)==18
    assert counts['TRAIN']=={'positive':180,'negative':90,'total':270}
    assert counts['QUALIFICATION']=={'positive':72,'negative':54,'total':126}
    assert counts['FINAL']=={'positive':108,'negative':72,'total':180}
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    bundle=build();counts=validate(bundle)
    if args.check_only:print(json.dumps(counts));return
    args.out.mkdir(parents=True,exist_ok=False)
    files={}
    for name,value in bundle.items():
        raw=encoded(value);(args.out/name).write_bytes(raw);files[name]={'sha256':digest(raw),'bytes':len(raw)}
    manifest={'schema':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':files,'counts':counts,
              'source_sha256':digest(Path(__file__).read_bytes()),'plan_sha256':digest(Path(__file__).with_name('PLAN.md').read_bytes()),
              'independence':'Authored without reading proposed router code or predictions; existing task metadata and prior six diagnostic questions only.'}
    (args.out/'MANIFEST.json').write_bytes(encoded(manifest))
    print(json.dumps({'out':str(args.out),'manifest_sha256':digest(encoded(manifest)),'counts':counts}))


if __name__=='__main__':main()
