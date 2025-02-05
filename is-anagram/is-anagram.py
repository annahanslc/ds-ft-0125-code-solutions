# %% [markdown]
# # Is-Anagram
#
# Exercise
# Write a function named “is_anagram” that takes two parameters, a string named “str_1” and a string named “str_2”. The function should return True if the words are anagrams of one another, and False if they are not.
#
# Examples: str_1 = “cautioned” str_1 = “cat” str_2 = “education” str_2 = “rat” Output: True Output: False

# %% [markdown]
# ### Method 1a: using dictionaries to keep count of characters and then compare dictionaries

# %%
def is_anagram(str_1, str_2):
  # Count the characters in the first string
  str_1_count = {}
  for character in str_1:
    if character in str_1_count.keys():
      str_1_count[character] = (str_1_count[character] + 1)
    else: str_1_count[character] = 1

  # Count the characters in the second string
  str_2_count = {}
  for character in str_2:
    if character in str_2_count.keys():
      str_2_count[character] = str_2_count[character] + 1
    else: str_2_count[character] = 1

  # Compare the two dictionaries
  for key in str_1_count.keys():
    if key in str_1_count and key in str_2_count:
      if (str_1_count[key] == str_2_count[key]):
        True
      else:
        return False
    else:
      return False
  return True

# %%
print(is_anagram('cautioned','education'))
print(is_anagram('cat','rat'))

# %% [markdown]
# ### Method 1b: condensed version of method 1a

# %%
def is_anagram_b(str_1, str_2):
  # Count the characters in the first string
  str_1_count = {}
  for character in str_1:
    str_1_count[character] = str_1_count.get(character,0) + 1

  # Count the characters in the second string
  str_2_count = {}
  for character in str_2:
    str_2_count[character] = str_2_count.get(character,0) + 1

  # Compare the two dictionaries
  return str_1_count == str_2_count

# %%
print(is_anagram_b('cautioned','education'))
print(is_anagram_b('cat','rat'))

# %% [markdown]
# ### Method 2: sorting the two strings and then comparing them

# %%
def is_anagram_2(str_1, str_2):
  return sorted(str_1) == sorted(str_2)

# %%
print(is_anagram_2('cautioned','education'))
print(is_anagram_2('cat','rat'))
