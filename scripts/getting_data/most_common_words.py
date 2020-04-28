#!/usr/bin/env python

import sys
from collections import Counter

# Pass in number of words as first argument
try:
    num_words = int(sys.argv[1])
except:
    print("Usage: most_common_words.py num_words")
    sys.exit(1) # Nonzero exit code indicates error

counter = Counter(word.lower() # Lowercase words
                  for line in sys.stdin
                  for word in line.strip().split() # Split on spaces
                  if word) # Skip empty "words"

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")
