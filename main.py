# Order orphans.json by alphabetical order of orphan name

# Currently on Kainuk

import json

with open("orphans.json", "r") as f:
    orphans = json.load(f)

# Sort orphans by name
orphans = dict(sorted(orphans.items(), key=lambda x: x[0]))

print(orphans)

def get_progress(orphan_name: str) -> int:
    """
    Given an orphan name, return how many images, up to and including that orphan (in the alphebetized list), exist using the orphans dict
    """
    return sum(orphans[key] for key in orphans.keys() if key <= orphan_name)

def get_length():
    return sum(orphans.values())

print(get_length())
print(get_progress("kainuk"))