-- =====================================================================
-- Customer Analytics SQL Queries (SQLite)
--
-- Tables (created by run_pipeline.py in data/processed/retail_analytics.db):
--   transactions(Invoice, StockCode, Description, Quantity, InvoiceDate,
--                UnitPrice, CustomerID, Country, TotalAmount)
--   customers(CustomerID, Recency, Frequency, Monetary, AvgOrderValue,
--             Country, R_Score, F_Score, M_Score, RFM_Score, Cluster,
--             Segment, PC1, PC2, ...)
--
-- Each query starts with a "-- name:" line so Python can load it by name.
-- Run all queries with:  python -m src.database
-- =====================================================================


-- name: total_revenue
-- Total revenue across all cleaned transactions.
SELECT ROUND(SUM(TotalAmount), 2) AS total_revenue
FROM transactions;


-- name: revenue_by_country
-- Revenue, customers and orders per country, highest revenue first.
SELECT Country,
       ROUND(SUM(TotalAmount), 2)   AS revenue,
       COUNT(DISTINCT CustomerID)   AS customers,
       COUNT(DISTINCT Invoice)      AS orders
FROM transactions
GROUP BY Country
ORDER BY revenue DESC
LIMIT 15;


-- name: top_customers
-- The 10 customers who spent the most.
SELECT CustomerID,
       Country,
       COUNT(DISTINCT Invoice)     AS orders,
       ROUND(SUM(TotalAmount), 2)  AS total_spent
FROM transactions
GROUP BY CustomerID, Country
ORDER BY total_spent DESC
LIMIT 10;


-- name: number_of_orders
-- Distinct orders (invoices), customers and products.
SELECT COUNT(DISTINCT Invoice)     AS total_orders,
       COUNT(DISTINCT CustomerID)  AS total_customers,
       COUNT(DISTINCT StockCode)   AS distinct_products
FROM transactions;


-- name: average_order_value
-- Average order value = total revenue / number of orders.
-- The inner query first adds up each order, then the outer query averages them.
SELECT COUNT(*)                     AS orders,
       ROUND(AVG(order_value), 2)   AS average_order_value,
       ROUND(MIN(order_value), 2)   AS smallest_order,
       ROUND(MAX(order_value), 2)   AS largest_order
FROM (
    SELECT Invoice, SUM(TotalAmount) AS order_value
    FROM transactions
    GROUP BY Invoice
);


-- name: monthly_revenue
-- Revenue, orders and active customers per month.
-- strftime('%Y-%m', ...) extracts year-month from the ISO date text.
SELECT strftime('%Y-%m', InvoiceDate)  AS month,
       ROUND(SUM(TotalAmount), 2)      AS revenue,
       COUNT(DISTINCT Invoice)         AS orders,
       COUNT(DISTINCT CustomerID)      AS active_customers
FROM transactions
GROUP BY month
ORDER BY month;


-- name: customer_purchase_frequency
-- How many customers placed 1 order, 2-3 orders, and so on.
-- A CTE (WITH ...) first counts orders per customer, then groups them into bands.
WITH orders_per_customer AS (
    SELECT CustomerID, COUNT(DISTINCT Invoice) AS orders
    FROM transactions
    GROUP BY CustomerID
)
SELECT CASE
           WHEN orders = 1 THEN '1 order'
           WHEN orders BETWEEN 2 AND 3 THEN '2-3 orders'
           WHEN orders BETWEEN 4 AND 10 THEN '4-10 orders'
           ELSE '11+ orders'
       END                              AS frequency_band,
       COUNT(*)                         AS customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders_per_customer), 1) AS pct_customers
FROM orders_per_customer
GROUP BY frequency_band
ORDER BY MIN(orders);


-- name: customer_monetary_value
-- Customers grouped into spending bands, with each band's share of revenue.
WITH spend_per_customer AS (
    SELECT CustomerID, SUM(TotalAmount) AS total_spent
    FROM transactions
    GROUP BY CustomerID
)
SELECT CASE
           WHEN total_spent < 250 THEN 'Under 250'
           WHEN total_spent < 1000 THEN '250 - 999'
           WHEN total_spent < 5000 THEN '1,000 - 4,999'
           ELSE '5,000 and above'
       END                                   AS spend_band,
       COUNT(*)                              AS customers,
       ROUND(SUM(total_spent), 2)            AS revenue,
       ROUND(100.0 * SUM(total_spent) / (SELECT SUM(total_spent) FROM spend_per_customer), 1)
                                             AS pct_revenue
FROM spend_per_customer
GROUP BY spend_band
ORDER BY MIN(total_spent);


-- name: segment_summary
-- Segment-level summary from the clustering results (customers table).
SELECT Segment,
       COUNT(*)                      AS customers,
       ROUND(AVG(Recency), 1)        AS avg_recency_days,
       ROUND(AVG(Frequency), 1)      AS avg_orders,
       ROUND(AVG(Monetary), 2)       AS avg_spend,
       ROUND(SUM(Monetary), 2)       AS total_revenue
FROM customers
GROUP BY Segment
ORDER BY total_revenue DESC;
