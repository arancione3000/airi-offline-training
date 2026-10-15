"""Ephemeral integration check on an actual AIRI release, never published as learning."""
import gc
import json
import random
import tempfile
from pathlib import Path
import torch
from generalist_lm.runtime import GeneralistRuntime
from session import train_step, save_session, restore_session
from success_bank import empty_bank, validate_bank, summary, row_key
from success_transfer import harvest, distill_step
from morph_model import new_state,consolidate,model_record
from chat_current import validate_selected_model


def real_checkpoint_smoke(donor, optimizer, checkpoint, eligible, encode, anchors, fingerprint, state_digest):
    """Exercise real weights, mining, distillation and committed bank restoration."""
    champion = GeneralistRuntime.from_checkpoint(checkpoint/'model')
    champion.model.eval()
    for parameter in champion.model.parameters(): parameter.requires_grad_(False)
    champion_digest = state_digest(champion.model)
    # A short known solver answer, selected from training independently of eval.
    candidates = sorted(eligible.values(), key=lambda r: (len(r['expected'].encode()), row_key(r)))[:8]
    row = candidates[0]; batch = encode(row); anchor = anchors[0]
    for group in optimizer.param_groups: group['lr'] = 1e-5
    for _ in range(24):
        train_step(donor.model,champion.model,optimizer,*batch,*anchor,kl_weight=.5)
    bank = empty_bank(fingerprint)
    harvest(donor,champion,candidates,encode,bank,eligible,
            {'step':24,'fingerprint':fingerprint,'model_sha256':state_digest(donor.model),
             'runner_sha':'ephemeral-integration-check'},generation_limit=2)
    validate_bank(bank,eligible,fingerprint,donor.config.vocab_size)
    if not bank['entries']:
        raise AssertionError('actual native candidate produced no verifiable token donation')
    del donor,optimizer; gc.collect()
    student = GeneralistRuntime.from_checkpoint(checkpoint/'model')
    optimizer = torch.optim.AdamW(student.model.parameters(),lr=1e-5)
    lesson = next(iter(bank['entries'].values()))
    losses=[]
    for _ in range(8):
        stats=distill_step(student.model,champion.model,optimizer,lesson,encode(lesson['row']),anchor,
                           lr_multiplier=1.,lr_cap=1e-5)
        losses.append(stats['loss']); bank['transfer_updates']+=1
    if not losses[-1] < losses[0]:
        raise AssertionError('actual native selective transfer loss did not decrease')
    if state_digest(champion.model) != champion_digest or any(p.grad is not None for p in champion.model.parameters()):
        raise AssertionError('integration check mutated protected champion')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'saved'; rng=random.Random(1); anchor_rng=random.Random(2)
        save_session(student,optimizer,rng,anchor_rng,{'completed_steps':32},path,
                     fingerprint=fingerprint,success_bank=bank)
        restored,_,_=restore_session(path,lambda m:torch.optim.AdamW(m.parameters(),lr=1e-5),fingerprint=fingerprint)
        recovered=json.loads((path/'success_bank.json').read_text())
        validate_bank(recovered,eligible,fingerprint,student.config.vocab_size)
        if recovered != bank or state_digest(restored.model) != state_digest(student.model):
            raise AssertionError('native bank/checkpoint restart mismatch')
        # Validate the stable reader on the exact native model files, including
        # its configuration; this test never writes the public model channel.
        manifest=json.loads((path/'checkpoint.json').read_text())
        ledger=new_state(state_digest(student.model))
        record=model_record(ledger,release='airi-offline-32-smoke-1-1',fingerprint=fingerprint,step=32,
                            model_files={n:d for n,d in manifest['files'].items() if n.startswith('model/')})
        validate_selected_model(path,record)
        before_transaction=state_digest(student.model)
        def learn(entry):
            distill_step(student.model,champion.model,optimizer,entry,encode(entry['row']),anchor,
                         lr_multiplier=1.,lr_cap=1e-5)
        def restore_transaction():
            nonlocal student,optimizer
            student,optimizer,_=restore_session(path,lambda m:torch.optim.AdamW(m.parameters(),lr=1e-5),fingerprint=fingerprint)
        result=consolidate(ledger,bank,{},train=learn,evaluate=lambda:{},restore=restore_transaction,
                           can_train=lambda:True,can_adopt=lambda before,after:False,
                           step=lambda:36,max_updates=4)
        if result['action']!='skill_rejected' or state_digest(student.model)!=before_transaction:
            raise AssertionError('actual native skill transaction failed to preserve prior weights')
        if ledger['attempted_updates']!=4:
            raise AssertionError('discarded native updates disappeared from compute ledger')
    print(json.dumps({'phase':'real native selective transfer integration PASS',
                      'checkpoint':str(checkpoint),'parameters':sum(p.numel() for p in student.model.parameters()),
                      'initial_transfer_loss':losses[0],'final_transfer_loss':losses[-1],
                      'bank':summary(bank),'champion_unchanged':True,'optimizer_updates':36,
                      'single_model_identity':record['model_id'],'transaction_rollback_verified':True,
                      'stable_reader_integrity_verified':True,
                      'scope':'ephemeral gradient and restart check; not evidence of conversational improvement',
                      'live_promoted':False}),flush=True)
