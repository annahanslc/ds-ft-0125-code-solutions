-- sql-project-2

-- 1. Create a temp table to store the row quantity of each table in “loandb” and the temp table includes 2 columns, 
-- one is “table_name” and the other is “row_quantity.” Show the table in the end. 
-- After take a screenshot of the result, then, drop the table.

CREATE TEMPORARY TABLE dsstudent.temp_row_qty
	(table_name varchar(30), row_quantity bigint);

INSERT INTO dsstudent.temp_row_qty
	(table_name, row_quantity)
	SELECT "train", COUNT(*) FROM train
	UNION ALL
	SELECT "bureau", COUNT(*) FROM bureau
	UNION ALL
	SELECT "bureau_balance", COUNT(*) FROM bureau_balance;
 	UNION ALL	
	SELECT "previous_application", COUNT(*) FROM previous_application
	UNION ALL
	SELECT "installments_payments", COUNT(*) FROM installments_payments
	UNION ALL
	SELECT "POS_CASH_balance", COUNT(*) FROM POS_CASH_balance
	UNION ALL
	SELECT "credit_card_balance", COUNT(*) FROM credit_card_balance;

SELECT *
FROM dsstudent.temp_row_qty;

DROP TABLE dsstudent.temp_row_qty;


-- 2. Show the monthly and annual income

SELECT AMT_INCOME_TOTAL annual_income, AMT_INCOME_TOTAL/12 monthly_income
FROM train;

-- 3. Transform the “DAYS_BIRTH” column by dividing “-365” and round the value to the integer place. 
-- Call this column as “age.”

SELECT ROUND(DAYS_BIRTH/-365) age
FROM train;

-- 4. Show the quantity of each occupation type and sort the quantity in descending order.

SELECT OCCUPATION_TYPE, COUNT(OCCUPATION_TYPE) as quantity
FROM train
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE
ORDER BY quantity DESC;


-- 5. In the field “DAYS_EMPLOYED”, the maximum value in this field is bad data, 
-- can you write a conditional logic to mark these bad data as “bad data”, 
-- and other values are “normal data” in a new field called “Flag_for_bad_data”?

SELECT DAYS_EMPLOYED,
	CASE
		WHEN DAYS_EMPLOYED = (SELECT MAX(DAYS_EMPLOYED) FROM train) THEN "bad data"
		ELSE "normal data"
	END as Flag_for_bad_data
FROM train;


-- 6. Can you show the minimum and maximum values for both “DAYS_INSTALLMENT” & “DAYS_ENTRY_PAYMENT” 
-- fields in the “installment_payments” table for default v.s. non-default groups of clients?

WITH temp as(
	SELECT t.SK_ID_CURR, t.TARGET, ip.DAYS_INSTALMENT, ip.DAYS_ENTRY_PAYMENT
	FROM train t
	INNER JOIN installments_payments ip
	ON t.SK_ID_CURR = ip.SK_ID_CURR
)
SELECT TARGET, MAX(DAYS_INSTALMENT), MIN(DAYS_INSTALMENT), MAX(DAYS_ENTRY_PAYMENT), MIN(DAYS_ENTRY_PAYMENT)
FROM temp
GROUP BY TARGET;
