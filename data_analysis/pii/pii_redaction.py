import json
import random
import string
import ipaddress


# List of random private IP addresses to use as replacements
REPLACEMENTS_IP = {
    "IPv4": ["172.16.31.10", "172.16.58.3", "192.168.127.12", "192.168.3.11"],
    "IPv6": [
        "fd00:c2b6:b24b:be67:2827:688d:e6a1:6a3b",
        "fc00:e968:6179::de52:7100",
        "fc00:db20:35b:7399::5",
        "fdf8:f53e:61e4::18",
    ],
}


def load_json(sample):
    try:
        return json.loads(sample)
    except ValueError:
        return []


def random_replacements():
    """Build dictionaries of random replacements for PII (key, email, IP address)

    Emails: replace with one of 4 [random string of 5 characters + @email.com]
    IP addresses: replace with one of 4 synthetic private IP addresses (IPv4 or IPv6)
    Keys: replace with one of 4 [sequence of 32 random characters/digits]

    TODO: add IPv6 and IPv4 separation
    """
    letters = string.ascii_lowercase
    lettters_digits = string.ascii_lowercase + string.digits
    emails = [
        "".join(random.choice(letters) for i in range(5)) + "@email.com"
        for i in range(4)
    ]
    keys = [
        "".join(random.choice(lettters_digits) for i in range(32)) for i in range(4)
    ]
    ip_addresses = REPLACEMENTS_IP
    return {"EMAIL": emails, "KEY": keys, "IP_ADDRESS": ip_addresses}


def replace_ip(value, replacements_dict):
    """Replace an IP address with a synthetic IP address of the same format"""
    try:
        ipaddress.IPv4Address(value)
        return random.choice(replacements_dict["IP_ADDRESS"]["IPv4"])
    except ValueError:
        try:
            ipaddress.IPv6Address(value)
            return random.choice(replacements_dict["IP_ADDRESS"]["IPv6"])
        except ValueError:
            # this doesn't happen if we already use ipaddress filter in the detection
            print("Invalid IP address")
            return value


def is_private_ip(ip):
    """Check if an IP address is allocated for private networks"""
    ip = ipaddress.ip_address(ip)
    return ip.is_private


def redact_pii_text(text, secrets, replacements, add_references=False):
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
    secrets = load_json(secrets)
    if secrets:
        secrets = sorted(secrets, key=lambda x: x["start"])
        # store the secrets that were replaced here with their replacements
        replaced_secrets = {}
        subparts = []
        references = []
        step = 0
        last_text = text
        for secret in secrets:
            # skip if the secret is an IP address allocated for private networks
            if secret["tag"] == "IP_ADDRESS" and is_private_ip(secret["value"]):
                continue
            subtext = text[step : secret["start"]]
            subpart = subtext if subtext else " "
            subparts.append(subpart)
            # if secret is already in replaced_secrets, use the same replacement
            if secret["value"] in replaced_secrets:
                replacement = replaced_secrets[secret["value"]]
            else:
                if secret["tag"] == "IP_ADDRESS":
                    replacement = replace_ip(secret["value"], replacements)
                else:
                    replacement = random.choice(replacements[secret["tag"]])
                replaced_secrets[secret["value"]] = replacement
            subparts.append(replacement)
            replaced_secrets[secret["value"]] = replacement
            if add_references:
                references.append(subpart)
                references.append(f"PI:{secret['tag']}:{replacement}END_PI")
            last_text = text[secret["end"] :]
            step = secret["end"]
        # if supbarpts are not empty join them (it can be empty when all secrets were skipped)
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
    for text, secrets, has_secrets, id in zip(
        examples["content"], examples["secrets"], examples["has_secrets"], examples["id"]
    ):
        if has_secrets:
            result = redact_pii_text(
                text, secrets, replacements=replacements, add_references=True
            )
            new_contents.append(result[0])
            references.append(result[1])
        else:
            new_contents.append(text)
            references.append(text)
    return {"new_content": new_contents, "redaction_refs": references}
