import os
import tempfile
from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings


def scan_str_for_secrets(text):
    '''Write the text to a temporary file and scan it for secrets.'''
    fp = tempfile.NamedTemporaryFile(delete=False, mode='w')
    fp.write(text)
    fp.close()
    secrets = SecretsCollection()
    with default_settings():
        secrets.scan_file(fp.name)
    os.unlink(fp.name)
    secrets_dict = list(secrets.data.values())
    result = []
    if secrets_dict:
        for secret in secrets_dict[0]:
            result.append({
                'type': secret.type,
                'line_number': secret.line_number,
                'secret_value': secret.secret_value,
            })

    return result