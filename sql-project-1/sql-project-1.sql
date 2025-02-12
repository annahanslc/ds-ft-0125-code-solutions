-- sql-project-1

-- 1. In the ‘dsstudent’ database, create a permanent table named “customer_{your_name}.”
CREATE TABLE customer_anna
	(customer_id smallint, name varchar(20), location varchar(20), total_expenditure varchar(20));

-- 2. Insert the following records to the “customer_{your_name}” table:
INSERT INTO customer_anna
	(customer_id, name, location, total_expenditure)
VALUES
	(1701, "John", "Newport Beach, CA", 2000),
	(1707, "Tracy", "Irvine, CA", 1500),
	(1711, "Daniel", "Newport Beach, CA", 2500),
	(1703, "Ella", "Santa Ana, CA", 1800),
	(1708, "Mel", "Orange, CA", 1700),
	(1716, "Steve", "Irvine, CA", 18000);

-- 3. Oops! The value in the field ”total_expenditure” of Steve is not correct. It should be “1800.” 
-- Can you update this record?
UPDATE customer_anna
SET total_expenditure = 1800
WHERE name = "Steve";

-- 4. We would like to update our customer data. Can you insert a new column called “gender” in the “customer_{your_name}” table?
ALTER TABLE customer_anna
ADD gender VARCHAR(20);

-- 5. Then, update the field “gender” with the following records:
UPDATE customer_anna
SET gender = CASE
	WHEN name = 'John' THEN 'M'
	WHEN name = 'Tracy' THEN 'F'
	WHEN name = 'Daniel' THEN 'M'
	WHEN name = 'Ella' THEN 'F'
	WHEN name = 'Mel' THEN 'F'
	WHEN name = 'Steve' THEN 'M'
END 
WHERE name IN ('John','Tracy','Daniel','Ella','Mel','Steve');

-- 6. The customer, Steve, decides to quit our membership program, so delete his record from the “customer_{your_name}” table.
DELETE FROM customer_anna
WHERE name = 'Steve';

-- 7. Add a new column called “store” in the table “customer_{your_name}”
ALTER TABLE customer_anna
ADD store varchar(20);

-- 8. Then, delete the column called “store” in the table “customer_{your_name}” because you accidentally added it.
ALTER TABLE customer_anna
DROP COLUMN store;

-- 9. Use “SELECT” & “FROM” to query the whole table “customer_{your_name}”
SELECT *
FROM customer_anna;

-- 10. Return “name” and “total_expenditure” fields from the table “customer_{your_name}”
SELECT name, total_expenditure
FROM customer_anna;

-- 11. Return “name” and “total_expenditure” fields from the table “customer_{your_name}” by using column alias (“AS” keyword)
SELECT name AS n, total_expenditure AS total_exp
FROM customer_anna;

-- 12. Change the datatype of the field “total_expenditure” from “VARCHAR” to ”SMALLINT”
ALTER TABLE customer_anna
MODIFY COLUMN total_expenditure SMALLINT;

-- 13. Sort the field “total_expenditure” in descending order
SELECT total_expenditure
FROM customer_anna
ORDER BY total_expenditure DESC;

-- 14. Return the top 3 customer names with the highest expenditure amount from the table “customer_{your_name}”
SELECT name, total_expenditure
FROM customer_anna
ORDER BY total_expenditure DESC
LIMIT 3;

-- 15. Return the number of unique values of the field “location” and use the column alias to name the return field as “nuniques”
SELECT COUNT(DISTINCT(location)) as nuniques
FROM customer_anna;

-- 16. Return the unique values of the field “location” and use the column alias to name the return field as “unique_cities”
SELECT DISTINCT(location) as unique_cities
FROM customer_anna;

-- 17. Return the data where the gender is male.
SELECT *
FROM customer_anna
WHERE gender = "M";

-- 18. Return the data where the gender is female.
SELECT *
FROM customer_anna
WHERE gender = "F";

-- 19. Return the data where the location is “Irvine, CA”
SELECT *
FROM customer_anna
WHERE location = "Irvine, CA";

-- 20. Return “name” and “location” where the 
-- ”total_expenditure” is less than 2000 and sort the result by the field “name” in ascending order
SELECT name, location
FROM customer_anna
WHERE total_expenditure < 2000
ORDER BY name;

-- 21. Drop the table “customer_{your_name}” after you finish all the questions.
DROP TABLE customer_anna;

