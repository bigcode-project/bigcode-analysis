import os
import tempfile
from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings


def scan_secrets(text):
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
                'start_index': text.index(secret.secret_value),
                'end_index': text.index(secret.secret_value) + len(secret.secret_value),
            })

    return result


def scan_secrets_batch(examples):
    """Scan a batch of examples from a dataset for secret keys
    This add two columns to the dataset:
    - pii: (list) of secrets found
    - has_pii: (bool) whether the example contains secret
    """

    list_secrets = []
    list_types = []
    list_limits = []
    has_secrets = []
    for text in examples["content"]:
        output = scan_secrets(text)
        if  output:
            # get secret values of each element in output
            # to add this in datasets we need same number of samples in each row
            # we save it as str instead of list
            secrets = str([e['secret_value'] for e in output])
            types = str([e['type'] for e in output])
            limits = str([(e['start_index'], e['end_index']) for e in output])
            list_secrets.append(secrets)
            list_types.append(types)
            list_limits.append(limits)
            has_secrets.append(True)
        else:
            list_secrets.append("")
            list_types.append("")
            list_limits.append("")
            has_secrets.append(False)
    return {"secrets": list_secrets, "types": list_types, "has_secrets": has_secrets}


def scan_secrets_batch_viz(examples):
    outputs = []
    for i, text in enumerate(examples["content"]):
        output = scan_secrets(text, suffix=".txt")
        if  output:
            outputs.append({"id": i, "secrets": output})
    return outputs