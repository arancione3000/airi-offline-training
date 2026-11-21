"""Run a single opt-in Airi request with a verified research receipt."""
import argparse
import json
from pathlib import Path

from chat_adapter import make_assistant
from learn import BUNDLED
from skill_bank import SkillBank
from transfer import MAX_RECEIPT_BYTES, verify


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--receipt', type=Path, required=True)
    cli.add_argument('--trusted-evidence-sha256', required=True)
    cli.add_argument('--messages', type=Path, required=True)
    args = cli.parse_args()
    try:
        with args.receipt.open('rb') as stream:
            certified = verify(stream.read(MAX_RECEIPT_BYTES + 1), args.trusted_evidence_sha256)
        messages = json.loads(args.messages.read_bytes())
    except (OSError, ValueError) as exc:
        cli.error(str(exc))
    assistant = make_assistant(lambda _: 'Nessuna procedura verificata disponibile.', certified,
                               bank=SkillBank(BUNDLED))
    print(json.dumps(assistant.respond(messages), ensure_ascii=False))


if __name__ == '__main__':
    main()
