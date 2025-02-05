# %% [markdown]
# # letters-to-symbols
#
# Exercise
#
# Write a function named “letters_to_symbols” that will take one parameter, a string named ‘s’. The function will encode the string so that the amount of letters will be represented by a number and the letter.
#
# Example: s = "AAAABBBCCDAAA" Output: "4A3B2C1D3A"

# %%
def letters_to_symbols(string):
  counter = 1
  result = []
  for n in range(len(string)-1):
    if string[n] == string[n+1]:
      counter += 1
    else:
      result.extend([str(counter),string[n]])
      counter = 1
  result.extend([str(counter),string[-1]])

  return ''.join(result)

# %%
s = "AAAABBBCCDAAA"
counted_string = letters_to_symbols(s)
print(counted_string)
