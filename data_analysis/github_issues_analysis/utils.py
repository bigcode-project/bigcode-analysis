import re

import datasets
import regex
import torch
from transformers import pipeline

GITHUB_EMAILS = [
    re.compile(pattern, re.DOTALL)
    for pattern in [
        "(.*)From:.+Reply to this email directly.+view it on GitHub(.*)\n?(.*)",
        "(.*)On.+notifications@github.com.+wrote:.+Reply to this email directly.+view it on GitHub(.*)\n?(.*)",
        "(.*)Signed-off-by: .+<.+>(.*?)\n?(.*)",
    ]
]
GITHUB_EMAIL_DATE = re.compile("\d+/\d+/\d+ \d{2}:\d{2} [AP]M.+wrote")
GITHUB_EMAIL_LINEBREAK = re.compile("_{20,}")


BOT_AUTHORS = [
    "Apache-HBase",
    "AutorestCI",
    "CLAassistant",
    "cmsbuild",
    "codecov-io",
    "codecov-commenter",
    "coveralls",
    "danger-public",
    "dnfclas",
    "msftclas",
    "PyDocTeur",
    "SparkQA",
    "karma-pr-reporter",
    "danger-public",
    "claassistantio",
    "probot-stale",
]

BOT_KEYWORDS = ["[bot]", "botmanager", "bors-", "jenkins", "k8s-", "-test-", "travis"]

BOT_SUFFIXES = [
    "-automaton",
    "-automation",
    "-benchmark",
    "-build",
    "-deployer",
    "-cloud",
    "bot",
    "-ci",
    "-linter",
    "-teamcity",
    "-test",
    "-testing",
    "-Service-Account",
]


def merge_text_columns(example):
    """Combines description and comment to one column (text)

    Descriptions are issue-level text (body of text when opening an issue),
    comments are replies to the parent issue or one of its comments.
    We merge them as an event cannot have both at the same time.
    """
    events_new = []
    text_columns = ["comment", "description"]
    for event_old in example["events"]:
        event_new = {k: v for k, v in event_old.items() if k not in text_columns}
        comment, description = event_old["comment"], event_old["description"]
        text = comment if comment else description
        event_new["text"] = text if text else ""
        events_new.append(event_new)
    example["events"] = events_new
    return example


def _strip_automated_email_text(text):
    """Removes text auto-generated when users post in issues via email reply"""
    if text:
        text = text.strip()
    else:
        return ""
    # try to extract with regex directly
    for pattern in GITHUB_EMAILS:
        m = pattern.match(text)
        if m:
            break
    if m:
        text = m.group(1) + m.group(3)
    else:
        # if no exact matches, apply matching line by line and
        # get potential content before/after automated email text
        lines = text.split("\n")
        start, end = 0, -1
        for i, line in enumerate(lines):
            line = line.strip()
            if "notifications@github.com" in line or bool(
                GITHUB_EMAIL_DATE.search(line)
            ):
                start = i
            if "Reply to this email directly" in line:
                end = i + 1 if line.endswith(":") else i
            if line.startswith(">"):
                # remove quoted text in replies
                end = i
        text = "\n".join(lines[:start] + lines[end + 1 :])
    # remove page break line
    return GITHUB_EMAIL_LINEBREAK.sub("", text).strip()


def strip_automated_email_text(example):
    """Removes auto-generated text from emails in Github issues"""
    # assumes merge_text_columns() was already applied on dataset
    example["events"] = [
        {
            k: _strip_automated_email_text(v) if k == "text" else v
            for k, v in event.items()
        }
        for event in example["events"]
    ]
    return example


def remove_bot_comments(example):
    """Discard auto comments from issues based on author pattern matching"""
    filtered_events = []
    modified = False
    for event in example["events"]:
        author = event["author"]
        # assumes single `text' field rather than comment/description
        is_bot = (
            any(bp.lower() in author.lower() for bp in BOT_KEYWORDS)
            or any(author.lower().endswith(s) for s in BOT_SUFFIXES)
            or any(author == a for a in BOT_AUTHORS)
        )
        if not is_bot:
            filtered_events.append(event)
        else:
            modified = True
    # example["old_events"] = example["events"]
    example["events"] = filtered_events
    example["bot_issue"] = len(example["events"]) == 0
    example["modified_by_bot"] = modified
    return example
