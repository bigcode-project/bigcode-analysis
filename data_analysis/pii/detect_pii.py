import json

from utils.emails_ip_addresses_detection import detect_email_addresses
from utils.keys_detection import detect_keys


def postprocess_secrets(secrets):
    """Postprocess the secrets found by the scan_secrets function
    """
    if secrets:
        matches = json.dumps(secrets)
        has_secrets = True
    else:
        matches = json.dumps([])
        has_secrets = False
    return matches, has_secrets


def scan_pii_batch(examples, key_detector="regex", new_email_regex=False):
    """Scan a batch of examples from a dataset for secret keys
    This add two columns to the dataset:
    - secrets: (list) of secrets found
    - has_secrets: (bool) whether the example contains secret
    """
    list_secrets = []
    list_has_secrets = []
    for text in examples["content"]:
        secrets = []
        if key_detector == "regex":
            # use a regex to detect keys + emails + ips
            secrets = secrets + detect_email_addresses(text, tag_types={"KEY", "EMAIL", "IP_ADDRESS"}, new_email_regex=new_email_regex)
        else:
            # for keys use detect-secrets tool
            secrets = secrets + detect_email_addresses(text, tag_types={"EMAIL", "IP_ADDRESS"}, new_email_regex=new_email_regex)
            # detect emails and ip addresses with regexes
            secrets = secrets + detect_keys(text)
        # to add this as new columns to datasets we need the same number of samples in each row
        # we save secrets as json strings instead of lists
        matches, has_secrets = postprocess_secrets(secrets)
        list_secrets.append(matches)
        list_has_secrets.append(has_secrets)
    return {"secrets": list_secrets, "has_secrets": list_has_secrets}


def scan_pii_batch_viz(examples, key_detector="regex", new_email_regex=False):
    """Scan a batch of examples from a dataset for secret keys
    and store results in lists for easy visualization"""
    secrets, outputs = [], []
    for i, text in enumerate(examples["content"]):
        if key_detector=="regex":
            # use a regex to detect keys + emails + ips
            secrets = secrets + detect_email_addresses(text, tag_types={"KEY", "EMAIL", "IP_ADDRESS"}, new_email_regex=new_email_regex)
        else:
            # for keys use detect-secrets tool
            secrets = secrets + detect_email_addresses(text, tag_types={"EMAIL", "IP_ADDRESS"}, new_email_regex=new_email_regex)
            # detect emails and ip addresses with regexes
            secrets = secrets + detect_keys(text)
        if  secrets:
            outputs.append({"index": i, "secrets": secrets})
    return outputs