# -*- coding: utf-8 -*-
"""MST BigScience PII Code

Original colab that is a source of this file is located at
    https://colab.research.google.com/drive/1086H3-LGMz3gX0pGy9ECgr8KflosSKso

# License

Copyright 2022 Authors of this Notebook

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Highest Risk
### Simple spans of characters:
# TODO improve key pattern regex
*   **Key [general]**:  API, SSH, GPG keys (we avoid removing short encodings and paths (with less than 8 "/"))
*   **Email address**, 
*   **IP address**: Digits with periods in them

### More complex spans: (WORK IN PROGRESS)
* **Full Names**: Requires additional NER package
* **Address**


## Lower Risk: (We're not doing)
*   **URL**
*   **Time**: dateparser dependency
*   **Date**: dateparser dependency
*   **Age**

"""


#@title Define highest risk PII.
TEXT_COLUMN = "content"
high_risk_tags = {'KEY', 'EMAIL', 'IP_ADDRESS'} 

"""# Regexes"""

#@title Get the less sophisticated MST regexes for High Risk scenarios (baseline comparison). Not language-specific; all are general.
import sys
import regex
# These are ordered so that we can return upon a match; no need to search for a substring.
year_patterns = [
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/][1-2][0-9]{3})(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # yyyy-yyyy or yyyy/yyyy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/.][0-3][0-9][\p{Pd}/.][0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # yyyy-mm-dd or yyyy-dd-mm or yyyy/mm/dd or yyyy/dd/mm or yyyy.mm.dd or yyyy.dd.mm
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/.][0-3][0-9][\p{Pd}/.](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # mm-dd-yyyy or dd-mm-yyyy or mm/dd/yyyy or dd/mm/yyyy or mm.dd.yyyy or dd.mm.yyyy or the same but with yy instead of yyyy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # mm-yyyy or mm/yyyy or the same but with yy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}-[0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # yyyy-mm or yyyy/mm
]

# Patterns for high-risk character strings
# bigscience old key regex: https://regex101.com/r/ZtYkjm/1 also detects paths
# new regex: https://regex101.com/r/YqruRP/1 paths with less than 10 '/' will not be detected, most ssh keys have many /
key_pattern = r'((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:_]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){10,})(?:$|[\b\s\p{Han}@?,!;:\'\")(.])'
ipv4_pattern = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'
ipv6_pattern = r'(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
ip_pattern = r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + r"|".join([ipv4_pattern, ipv6_pattern]) + ")(?:$|[\s@,?!;:\'\"(.\p{Han}])"
# bigscience old regex: https://regex101.com/r/OZdSUu/5 detects emails without domain name and also catches python version such as python@2.7
# new regex hhttps://regex101.com/r/Huvtzb/1
email_pattern = r'([^\s@,?!;:)(]+@[^,\s!?;,]{3,}[\.][^\s\b\'\"@,?!;:)(.]+)'
# https://regex101.com/r/mOqi1s/3
#user_pattern = r'(?:^|[\s@,?!;:\'\")(\p{Han}])(@[^\s@,?!;:\'\")(]{3,})'
# we remove use_pattern because it removes decorators


key_regex = regex.compile(key_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
ipv4_regex = regex.compile(ipv4_pattern)
ipv6_regex = regex.compile(ipv6_pattern)
ip_regex = regex.compile(ip_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
email_regex = regex.compile(email_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
#user_regex = regex.compile(user_pattern, flags=regex.MULTILINE) #, re.MULTILINE)


#sasha_regexes = copy.deepcopy(regex_rulebase)
mst_regexes = {}
for tag in high_risk_tags:
  #print(tag)
  if tag == 'KEY':
    mst_regexes['KEY'] = key_regex
  elif tag == 'IPv4':
    mst_regexes['IPv4'] = ipv4_regex
  elif tag == 'IPv6':
    mst_regexes['IPv6'] = ipv6_regex
  elif tag == 'IP_ADDRESS':
    mst_regexes['IP_ADDRESS'] = ip_regex
  elif tag == 'EMAIL':
    mst_regexes['EMAIL'] = email_regex
  else:
    sys.stderr.write('Dont have tag regex pattern for %s =(' % tag)

#print("MST regexes under examination are:")
#for tag, regx in mst_regexes.items():
  #print(tag, end=":\t")
  #print(regx)

"""# PI Detection and Redaction functions are defined here! """

#@title The detection functions and basic filtering functions are defined here.
# tag_type = {'KEY', 'EMAIL', 'IP_ADDRESS'}
# Choose whether to put this import before or after, depending on which you're testing. =)

def ip_has_digit(matched_str):
  """Checks to make sure the PII span is not just :: or whatever that may
  accidentally be picked up by making sure there are digits."""
  return any(map(str.isdigit, matched_str))

def matches_date_pattern(matched_str):
  # Screen out date false positives
  for year_regex in year_patterns:
    if year_regex.match(matched_str):
      return True
  return False

def is_website(matched_str):
  # TODO
  return False

def detect_pii(text, tag_types):
  matches = []
  for tag in tag_types:
    label_pattern = mst_regexes[tag]
    # !! regex.match happens here!!
    matches_tmp = label_pattern.finditer(text)
    for match in matches_tmp:
      # TODO: Why does this happen?
      if match.groups():
        if len(match.groups()) > 1 and match.groups()[1]:
          sys.stderr.write("Warning: Found substring matches in the main match.")
          #print(tag)
          #print(text)
          #print(match.groups())
        matched_str = match.groups()
        # print(matched_str)
        # Why does this happen?
        matched_str = matched_str[0]
        if matched_str:
          if tag in ["IP_ADDRESS"]:
            # Filter out false positive IPs
            if not ip_has_digit(matched_str):
              continue
          if tag == "IP_ADDRESS":
            # Filter out date false positives
            if matches_date_pattern(matched_str):
              continue
          # TODO: Implement
          # if tag in ["KEY"]:
          #  # TODO: implement
          #  if is_website(matched_str):
          #    continue
          matches += [(matched_str, match.span(), str(label_pattern), tag)]
  return matches


#@title Redaction function defined here.
def redact_pii(text, matches):
  """Takes a match as defined in the detect_pii function and redacts it from the full string, returning a <redacted text, metadata> tuple.
  we replace email and IP adresses with dummy examples that respect the format in case there is code that requires this structure (e.g
  example of a regex to retrieve emails) """
  redacted_str = text
  metadata = []
  for match in matches:
    matched_str = match[0]
    tag = match[3]
    if tag == "EMAIL":
      redact_tag = "dummy@email.com"

    elif tag == "IP_ADDRESS":
      redact_tag = "127.0.0.1"
    
    else:
      redact_tag = "PI:" + tag
    
    redacted_str = redacted_str.replace(matched_str, redact_tag)
    # Create the "metadata" as all of the information we had before redaction
    metadata += [(match)]
  return (redacted_str, metadata)

#@title General function to run the PII detection and redact it, saving everything else to metadata, is defined here.
def run_pii(text):
  """
  Runs the given set of regexes on the data "lines" and pulls out the
  tagged items.
  """

  #print('Detecting....')
  # What is this for...?
  text = text.encode().decode()
  matches = detect_pii(text, high_risk_tags)
  #print(matches)
  match_set = (text, {})
  if len(matches) > 0:
    # !!! REDACTION HAPPENS HERE !!!
    redacted_str, metadata = redact_pii(text, matches)
    metadata_out = {"regex metadata":metadata, "original": text, "redacted": redacted_str}
    match_set = (redacted_str, metadata_out)
  return match_set


def run_pii_batch(exs):
    """
    Runs the given set of regexes on the data "lines" and pulls out the
    tagged items.
    The lines structure stores the language type(s). This can be used for
    language-specific regexes, although we're dropping that for now and using
    only "default"/non-language-specific regexes.
    """
    regex_metadata = []
    old_text = []
    new_text = []
    modified = []
    for text in exs[TEXT_COLUMN]:
        # What is this for...?
        text = text.encode().decode()
        matches = detect_pii(text, high_risk_tags)
        if len(matches) > 0:
            # !!! REDACTION HAPPENS HERE !!!
            redacted_str, metadata = redact_pii(text, matches)
            regex_metadata.append(repr(metadata))
            old_text.append(text)
            new_text.append(redacted_str)
            modified.append(True)
        else:
            regex_metadata.append("")
            old_text.append(text)
            new_text.append(text)
            modified.append(False)
    result = {
        "regex_metadata": regex_metadata,
        "old_text": old_text,
        TEXT_COLUMN: new_text,
        "modified": modified
    }
    return result