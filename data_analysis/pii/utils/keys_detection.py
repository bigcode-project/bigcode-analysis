import os
import tempfile

from detect_secrets import SecretsCollection
from detect_secrets.settings import transient_settings

# Secrets detection with detect-secrets tool


filters = [
    # some filters from [original list](https://github.com/Yelp/detect-secrets/blob/master/docs/filters.md#built-in-filters)
    # were removed based on their targets
    {"path": "detect_secrets.filters.heuristic.is_sequential_string"},
    {"path": "detect_secrets.filters.heuristic.is_potential_uuid"},
    {"path": "detect_secrets.filters.heuristic.is_likely_id_string"},
    {"path": "detect_secrets.filters.heuristic.is_templated_secret"},
    {"path": "detect_secrets.filters.heuristic.is_sequential_string"},
]
plugins = [
    {"name": "ArtifactoryDetector"},
    {"name": "AWSKeyDetector"},
    # the entropy detectors esp Base64 need the gibberish detector on top
    {"name": "Base64HighEntropyString"},
    {"name": "HexHighEntropyString"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "CloudantDetector"},
    {"name": "DiscordBotTokenDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "IbmCloudIamDetector"},
    {"name": "IbmCosHmacDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "MailchimpDetector"},
    {"name": "NpmDetector"},
    {"name": "SendGridDetector"},
    {"name": "SlackDetector"},
    {"name": "SoftlayerDetector"},
    {"name": "StripeDetector"},
    {"name": "TwilioKeyDetector"},
    # remove 3 plugins for keyword
    # {'name': 'BasicAuthDetector'},
    # {'name': 'KeywordDetector'},
    # {'name': 'PrivateKeyDetector'},
]


def detect_keys(content, suffix=".txt"):
    """Detect secret keys in content using detect-secrets tool
    Args:
        content (str): string containing the text to be analyzed.
        suffix (str): suffix of the file
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match."""

    fp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    fp.write(content)
    fp.close()
    secrets = SecretsCollection()
    with transient_settings(
        {"plugins_used": plugins, "filters_used": filters}
    ) as settings:
        secrets.scan_file(fp.name)
    os.unlink(fp.name)
    secrets_set = list(secrets.data.values())
    matches = []
    if secrets_set:
        for secret in secrets_set[0]:
            # TODO fix content.index: what if there are multiple occurences of the same key
            matches.append(
                {
                    "tag": "KEY", # secret.type
                    "value": secret.secret_value,
                    "start": content.index(secret.secret_value),
                    "end": content.index(secret.secret_value)
                    + len(secret.secret_value),
                }
            )
    return matches
