-- Average sales by store
SELECT
    Store,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY Store
ORDER BY avg_sales DESC;

-- Average sales by promotion status
SELECT
    Promo,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY Promo;

-- Average sales by store type
SELECT
    StoreType,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY StoreType
ORDER BY avg_sales DESC;

-- Average sales by month
SELECT
    Month,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY Month
ORDER BY Month;
