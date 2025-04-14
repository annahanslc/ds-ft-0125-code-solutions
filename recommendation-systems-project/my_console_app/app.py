
import pandas as pd
from recommenders import *

pd.set_option('display.max_columns', None)

def main():
  print("Welcome to the Google Local Business Recommender!")

  user_id = str(input("What is the user id? "))
  print()
  print(f'Hello, user number {user_id}')
  print()

  n_recs = int(input("How many recommendations would you like? "))
  print()
  print(f'Great, I will give you {n_recs} recommendations')
  print()

  print("Which type of recommender would you like to use?")
  print()
  print("1. User-based - suggests places liked by other users with similar tastes.")
  print("2. Item-based - suggests places similar to ones you have rated highly.")
  print("3. SVD-based - uses patterns in your past reviews to discover hidden preferences and suggest places that you might like.")
  print()
  type = str(input("Please enter 1, 2 or 3: "))

  if type == '1':
    print(f'Now compiling {n_recs} recommendations for user# {user_id},')
    print('...')
    print('using user-based collaborative filtering...')
    print('...')
    print('please wait a few minutes...')
    print('...')
    print(user_based_recommendations(user_id, n_recs))

  if type == '2':
    print(f'Now compiling {n_recs} recommendations for user# {user_id},')
    print('...')
    print('using item-based collaborative filtering...')
    print('...')
    print('please wait a few minutes...')
    print('...')
    print(item_based_recommendations(user_id, n_recs))

  if type == '3':
    print(f'Now compiling {n_recs} recommendations for user# {user_id},')
    print('...')
    print('using singular value decomposition (SVD)...')
    print('...')
    print('please wait a few minutes...')
    print('...')
    print(get_svd_recommendations(user_id, n_recs))

  print()
  again = str(input('Would you like to get more recommendations? Y/N  '))

  if again == 'Y':
    print()
    print()
    print()
    main()
  else:
    print()
    print()
    print()
    print('Goodbye!s')

if __name__ == "__main__":
  main()
