import os
import tempfile
from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings


def scan_str_content(content, suffix):
    fp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode='w')
    fp.write(content)
    fp.close()
    secrets = SecretsCollection()
    with default_settings():
        secrets.scan_file(fp.name)
    os.unlink(fp.name)
    secrets_set = list(secrets.data.values())
    result = []
    if secrets_set:
        for secret in secrets_set[0]:
            result.append({
                'type': secret.type,
                'line_number': secret.line_number,
                'secret_value': secret.secret_value,
            })

    return result