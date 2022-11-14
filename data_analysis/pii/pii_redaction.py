import json
import random
import string


def load_json(sample):
    try:
        return json.loads(sample)
    except ValueError:
        return []


def random_replacements():
    """Build dictionaries of random replacements for PII (key, email, IP address"""
    letters = string.ascii_lowercase
    lettters_digits = string.ascii_lowercase + string.digits
    generate_ip = lambda: ".".join([str(random.randint(0, 255)) for i in range(4)])
    generate_email = lambda x: "".join(x, "@email.com")
    # emails = ["dummuy@example.com", "example@email.com", "email@dummy.com"]
    emails = [
        "".join(random.choice(letters) for i in range(5)) + "@email.com"
        for i in range(4)
    ]
    keys = [
        "".join(random.choice(lettters_digits) for i in range(32)) for i in range(4)
    ]
    ip_addresses = [generate_ip() for i in range(4)]
    return {"EMAIL": emails, "KEY": keys, "IP_ADDRESS": ip_addresses}


def redact_pii_text(text, secrets, replacements=None, add_references=False):
    """Redact PII in a text
    Args:
        text (str): text to redact
        secrets (json): json string with the secrets to redact
        replacements (dict): dictionary of replacements for each PII type
        add_references (bool): whether to add references to the redacted text
        for vizualization
    Returns:
        text (str): new text with redacted secrets
    """
    if not replacements:
        replacements = random_replacements()
    secrets = load_json(secrets)
    if secrets:
        secrets = sorted(secrets, key=lambda x: x["start"])
        subparts = []
        references = []
        step = 0
        last_text = text
        for secret in secrets:
            localhost = ["127.0.0.1", "0:0:0:0:0:0:0:1", "::1"]
            # skip if secret is localhost or broadcast address
            if secret["tag"] == "IP_ADDRESS" and secret["value"].startswith(
                tuple(localhost)
            ):
                continue
            subtext = text[step : secret["start"]]
            subpart = subtext if subtext else " "
            subparts.append(subpart)
            # replace secret value
            replacement = random.choice(replacements[secret["tag"]])
            subparts.append(replacement)
            if add_references:
                references.append(subpart)
                references.append(f"PI:{secret['tag']}:{replacement}END_PI")
            last_text = text[secret["end"] :]
            step = secret["end"]
        # if supbarpts are not empty join them
        new_text = "".join(subparts) + last_text if subparts else last_text
        if add_references:
            references = "".join(references) + last_text if references else last_text
    else:
        new_text = text
        references = text
    result = (new_text, references) if add_references else new_text
    return result


def redact_pii_batch(examples, replacements=None):
    """Anonymize PII in a batch of examples from a dataset"""
    new_contents = []
    references = []
    for text, secrets, has_secrets in zip(
        examples["content"], examples["secrets"], examples["has_secrets"]
    ):
        if has_secrets:
            result = redact_pii_text(text, secrets, replacements, add_references=True)
            new_contents.append(result[0])
            references.append(result[1])
        else:
            new_contents.append(text)
            references.append(text)
    return {"new_content": new_contents, "redaction_refs": references}
