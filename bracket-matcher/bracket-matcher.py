# %% [markdown]
# # bracket-matcher
#
# Exercise
#
# Write a function named “BracketMatcher” that takes one parameter, a string named “s”. The function should return True if the brackets are correctly matched and each bracket is matched and False if they aren’t.
#
# Example: s = “(a((kl(mns)t)uvwz)” Output: False

# %%
s = "(a((kl(mns)t)uvwz)"     # The test example, result should be False
s2 = "(a((kl(mns)t)uvwz))"   # Second test example, result should be True
s3 = "(a))()kl(mnstuvwz"     # Third test example, number of brackets match, but in the wrong order, result should be False

def BracketMatcher(string):
  counter = 0                 # Start a counter to keep track of number of (
  for character in string:
    if character == '(':      # For every ( add 1 to counter
      counter += 1
    if character == ')':
      counter -= 1            # For every ) minus 1 to counter
    if counter < 0:
      return False            # Edge case, if there are more ) than (, then they cannot be correctly matched
  return counter == 0         # If no edge case occurred, and the number of ( & ) match, then return True

# %%
print(BracketMatcher(s))
print(BracketMatcher(s2))
print(BracketMatcher(s3))

# %%
