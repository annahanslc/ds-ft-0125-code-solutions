# %% [markdown]
# # visualization-project-laptop-sales

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
df = pd.read_csv('Sales_data.txt')
df.head()

# %%
df.describe()

# %% [markdown]
# # 1. What is target market? Male or female?
# Hint: consider total profit

# %%
profit_by_gender = df.groupby('Contact Sex')['Profit'].sum()
profit_by_gender

# %%
avg_profit_by_gender = df.groupby('Contact Sex')['Profit'].mean()
avg_profit_by_gender

# %%
gender_count = df.groupby('Contact Sex')['Contact Sex'].count()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

ax1.bar(profit_by_gender.index, profit_by_gender.values)
ax1.set_title('Total Profit By Gender')

ax2.bar(avg_profit_by_gender.index, avg_profit_by_gender.values)
ax2.set_title('Average Profit By Gender')

ax3.bar(gender_count.index, gender_count.values)
ax3.set_title('Gender in Sample')

plt.tight_layout()

# %% [markdown]
# Answer --> Based on the total profit, the target market should be Male. In the sample provided, total profit from Male customers exceed that of Female customers. However, there are also more Male customers than Female customers in the sample. When examining the average profit for each gender, Female customers provide a higher profit on average than Male customers. This means that although Male customers are currently the target market, by targeting more Female customers, we could potentially increase profitability. I would recommend targeting Female customers, unless the ability to acquire new Female customers is more costly than that of acquiring Male customers. In which case, further analysis should be conducted to evaluate the return on marketing cost to determine which gender should be targeted.

# %% [markdown]
# # 2. If business is cash-constrained, which gender should be targeted?
# Hint: business being cash-constrained means cost to business should be considered

# %%
avg_cost_by_gender = df.groupby('Contact Sex')['Our Cost'].mean()
avg_cost_by_gender

# %%
plt.bar(avg_cost_by_gender.index, avg_cost_by_gender.values)
plt.title("Average Cost By Gender");

# %% [markdown]
# Answer --> The average cost of a customer by gender shows that on average, sales to Female customers costs more than sales to Male customers. This means that if the company is under cash-constraints, and wishes to make sales at lower costs, then Male customers should be targeted.

# %% [markdown]
# # 3. If consumer is cash-constrained, which gender should be targeted?

# %%
average_price_by_gender = df.groupby('Contact Sex')['Sale Price'].mean()
average_price_by_gender

# %%
plt.bar(average_price_by_gender.index, average_price_by_gender.values)
plt.title('Average Price By Gender')

# %% [markdown]
# Answer --> If consumers are cash-constrained, then we should target Male customers. This is because, on average, the sale price to Male customers is lower than to Female customers. At a lower price, more customers would be likely to make a purchase. This means, that selling to Male customers is more likely to result in a sale.

# %% [markdown]
# # 4. What is our target age to maximize profit?
# Hint: consider individual ages and age ranges

# %%
fig, axes = plt.subplots(2,2)

axes[1,0].scatter(total_profit_by_age.index, total_profit_by_age.values)
axes[1,0].set_title("Total Profits By Customer Age")

axes[0,0].hist(df['Contact Age']);
axes[0,0].set_title("Customer Age Distribution")

axes[0,1].scatter(df['Contact Age'],df['Profit'])
axes[0,1].set_title("Customer Profitability by Age")

axes[1,1].bar(total_profit_by_age.index, total_profit_by_age.values)
axes[1,1].set_title("Total Profits By Customer Age Bar")

plt.tight_layout()

# %%
total_profit_by_age_df = total_profit_by_age.to_frame()
total_profit_by_age_df = total_profit_by_age_df.reset_index()

# %%
df['Contact Age'].describe()

# %% [markdown]
# Answer --> The target age for maximizing profitability based on this sample data is 46, because this age provides the highest total profit. The target age range would be between 43 and 53, as 50% of the customers fall within this age range.

# %% [markdown]
# ### Calculating target age range in a second way:

# %% [markdown]
# ##### Separating the ages into buckets and totalling the profits for each bucket:
#
# ##### 1) Calculating the number of bins

# %%
bin_size = 5
num_bins = ((df['Contact Age'].max() - df['Contact Age'].min()) // bin_size ) + 1
num_bins


# %% [markdown]
# ##### 2) Looping over the dataframe to populate the bins

# %%
bins = {}
range_start = df['Contact Age'].min()
for n in range(0,num_bins):
  sum = 0
  for _, record in total_profit_by_age_df.iterrows():
    if record["Contact Age"] >= range_start and record["Contact Age"] < (range_start + bin_size ):
      sum += record["Profit"]
  bins[f"{range_start}-{range_start + bin_size -1}"] = sum
  range_start += bin_size

bins

# %% [markdown]
# ##### 3) Converting the resulting data into a Dataframe

# %%
bins_df = pd.DataFrame(list(bins.items()), columns=["Age Range", "Total Profit"])
bins_df

# %% [markdown]
# ##### 4) Plotting the results in a bar graph

# %%
plt.bar(bins_df['Age Range'],bins_df['Total Profit'])

# %% [markdown]
# Answer B --> The above plot is in line with our original recommendation of target age range being 43-53, however, we should consider extending the upper end of the range to 57 as 53-57 is the group that returns the highest profits.

# %% [markdown]
# # 5. Which product should we feature?

# %%
products_by_total_profit = df.groupby('Product ID')['Profit'].sum()
products_by_total_profit

# %%
plt.bar(products_by_total_profit.index, products_by_total_profit.values, width=0.5)
plt.xticks(fontsize=7)
plt.title("Total Profits By Product")
plt.xlabel("Product ID")
plt.ylabel("Total Profit")
plt.tight_layout;

# %%
products_by_avg_profit = df.groupby('Product ID')['Profit'].mean()
products_by_avg_profit

# %%
plt.bar(products_by_avg_profit.index, products_by_avg_profit.values, width=0.5)
plt.xticks(fontsize=7)
plt.title("Products by Avg Profit")
plt.xlabel("Product ID")
plt.ylabel("Total Profit")
plt.tight_layout;

# %% [markdown]
# Answer --> We should feature the GT13-0024, because this product has the highest average profitability. It also returns the second highest total profitability. This means that the product is both marketable and profitable.
#
# GA401IV can also be considered for additional featuring, as it currently returns the highest total profits. However, I believe that additional sales of GT13-0024 is more likely increase overall profit more than any other product.

# %% [markdown]
# # 6. What lead sources have worked in the past: website, flyer, or email?

# %%
lead_by_total_profit = df.groupby('Lead Source')['Profit'].sum()
plt.bar(lead_by_total_profit.index, lead_by_total_profit.values)

# %% [markdown]
# Answer --> It appears that the website has worked the most consistently as a lead source in the past. It seems that Flyers are sometimes effective, and sometimes not. However, this depends on how many copies of each flyer version was distributed. Emails also work, although considering the total profits, it is overshadowed by flyers and the website. Regardless, it is not a fair comparison without knowing how many emails were sent in total, in order to compare it to how many flyers were distributed. The costs of these lead sources should also be considered.

# %% [markdown]
# # 7. When is the best time to do email marketing?

# %% [markdown]
# ##### Step 1: Calculate the frequency of the sales by month

# %%
email_by_month = df[df['Lead Source'] == 'Email'].groupby('Sale Month').size()
email_by_month = email_by_month.reset_index()
email_by_month

# %% [markdown]
# ##### Step 2: In order to see the distribution across the entire year, create a dataframe with all the months

# %%
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
months_df = pd.DataFrame(data=months, columns=["Month"])

# %% [markdown]
# ##### Step 3: Merge the sales data with the months dataframe using a Left join, where month_lf is the origin df

# %%
merged_months = pd.merge(months_df, email_by_month, left_on="Month", right_on="Sale Month", how="left")
dropped_months = merged_months.drop(columns=['Sale Month'])
fillna_months = dropped_months.fillna(0)
cleaned_months = fillna_months.rename(columns={0:"Sales"})
cleaned_months

# %%
plt.bar(cleaned_months['Month'], cleaned_months['Sales'])
plt.title("Frequency of Sales by Month")
plt.xlabel("Month")
plt.ylabel("Number of Sales")
plt.xticks(fontsize=6)
plt.tight_layout()

# %% [markdown]
# Answer --> Based on the sample data, the best time to do email marketing is at year-end, especially in the month of November. However, due to limited data, I would recommend gathering additional data in order to conduct analysis that can recommend with more confidence.

# %%
