"""Read-only natural dialogue probe of the retained native checkpoint."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'stable-training'))
import torch
from generalist_lm.runtime import GeneralistRuntime
from session import sha256_file
import transport

SINGLES=[
('greeting','Ciao Airi, come va oggi?'),
('identity','Come ti chiami?'),
('listen','Ho avuto una brutta giornata. Ti va di ascoltarmi?'),
('chat','Mi piacerebbe parlare un po\u2019 con te. Di cosa possiamo parlare?'),
('hobby','Mi piacciono gli anime. Tu cosa mi chiedi per conoscermi meglio?'),
('clarification','Non ho capito la tua risposta. Puoi spiegarti meglio?'),
('thanks','Grazie per avermi aiutato!'),
('math','Quanto fa 2 + 2?'),
('fact','Quanti giorni ci sono in una settimana?'),
('definition','Spiegami in una frase che cosa è un gatto.'),
('english','Hello Airi, how are you today?'),
('english-listen','I had a difficult day. Can we talk about it?'),
]
DIALOGUES=[
('name-memory',['Mi chiamo Edoardo.','Come mi chiamo?']),
('color-memory',['Il mio colore preferito è rosso.','Qual è il mio colore preferito?']),
('object-memory',['Ho messo il quaderno nel cassetto.','Dove ho messo il quaderno?']),
('followup',['Mi piace disegnare paesaggi.','Di che cosa stavamo parlando?']),
]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    spec=transport.SPEC
    transport.SEED_TAG=spec['migration_source_release']
    transport.SOURCE_FINGERPRINTS={transport.SEED_TAG:spec['migration_source_fingerprint']}
    def verify(path,expected_fingerprint=spec['fingerprint']):
        manifest=json.loads((path/'checkpoint.json').read_text())
        if manifest['fingerprint']!=expected_fingerprint:raise ValueError('checkpoint fingerprint mismatch')
        if not spec['start_step']<=manifest['completed_steps']<=spec['max_steps']:raise ValueError('checkpoint step mismatch')
        if not manifest['files']:raise ValueError('empty checkpoint')
        for name,digest in manifest['files'].items():
            file=path/name
            if not file.resolve().is_relative_to(path.resolve()) or sha256_file(file)!=digest:raise ValueError('checkpoint digest mismatch')
        return manifest
    transport.verify_checkpoint=verify
    root=a.root.resolve();checkpoint=transport.load_release(a.release,root)
    fingerprint=transport.SOURCE_FINGERPRINTS.get(a.release,spec['fingerprint'])
    manifest=verify(checkpoint,fingerprint)
    torch.set_num_threads(2);runtime=GeneralistRuntime.from_checkpoint(checkpoint/'model');runtime.model.eval()
    answers=[]
    def answer(case,messages):
        tick=time.monotonic()
        with torch.inference_mode():response=runtime.chat(messages,max_new_tokens=96,temperature=0.)
        row=dict(case=case,messages=[dict(m) for m in messages],response=response,seconds=time.monotonic()-tick)
        answers.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        return response
    for case,prompt in SINGLES:answer(case,[dict(role='user',content=prompt)])
    for case,prompts in DIALOGUES:
        messages=[]
        for i,prompt in enumerate(prompts):
            messages.append(dict(role='user',content=prompt));response=answer(case+'-'+str(i+1),messages);messages.append(dict(role='assistant',content=response))
    verify(checkpoint,fingerprint)
    result=dict(checkpoint_release=a.release,completed_steps=manifest['completed_steps'],fingerprint=fingerprint,
                context_length=runtime.config.context_length,temperature=0.,max_new_tokens=96,training_performed=False,
                checkpoint_unchanged=True,live_promoted=False,scope='20 natural prompts including actual two-turn responses; qualitative review',answers=answers)
    output=root/f'AIRI_NATURAL_PROBE_{os.environ["GITHUB_RUN_ID"]}.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    transport.gh('release','upload',a.release,output,'--repo',os.environ['GITHUB_REPOSITORY'])
if __name__=='__main__':main()
