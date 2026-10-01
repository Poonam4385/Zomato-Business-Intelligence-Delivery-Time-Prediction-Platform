-- Business query bank corrected for the cleaned dataset.
-- Frozen-dataset queries use MAX(OrderDate), not CURRENT_DATE.

-- 1. Executive KPIs
SELECT
    COUNT(*) AS all_orders,
    COUNT(*) FILTER (WHERE "OrderStatus" IN ('Delivered','Delivered Late')) AS completed_orders,
    ROUND(SUM("FinalAmount") FILTER (WHERE "OrderStatus" IN ('Delivered','Delivered Late')),2) AS completed_revenue,
    ROUND(AVG("FinalAmount") FILTER (WHERE "OrderStatus" IN ('Delivered','Delivered Late')),2) AS average_order_value,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "OrderStatus"='Cancelled') / COUNT(*),2) AS cancellation_rate_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE "OrderStatus"='Delivered Late') /
      NULLIF(COUNT(*) FILTER (WHERE "OrderStatus" IN ('Delivered','Delivered Late')),0),2) AS late_rate_pct
FROM orders;

-- 2. Last six months relative to the dataset snapshot
WITH snapshot AS (SELECT MAX("OrderDate") AS max_date FROM orders)
SELECT o.*
FROM orders o CROSS JOIN snapshot s
WHERE o."OrderDate" >= s.max_date - INTERVAL '6 months';

-- 3. Monthly completed revenue and MoM growth
WITH m AS (
    SELECT DATE_TRUNC('month', "OrderDate") AS month,
           SUM("FinalAmount") AS revenue,
           COUNT(*) AS orders
    FROM orders
    WHERE "OrderStatus" IN ('Delivered','Delivered Late')
    GROUP BY 1
), x AS (
    SELECT *, LAG(revenue) OVER (ORDER BY month) AS prev_revenue FROM m
)
SELECT *, ROUND(100.0*(revenue-prev_revenue)/NULLIF(prev_revenue,0),2) AS mom_growth_pct
FROM x ORDER BY month;

-- 4. City performance uses restaurant city as the operational location
SELECT r."City",
       COUNT(*) FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')) AS completed_orders,
       ROUND(SUM(o."FinalAmount") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),2) AS revenue,
       ROUND(AVG(o."DeliveryTimeMinutes") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),2) AS avg_delivery_minutes
FROM orders o JOIN restaurants r ON r."RestaurantID"=o."RestaurantID"
GROUP BY r."City" ORDER BY revenue DESC;

-- 5. Top restaurants per city by completed revenue
WITH r AS (
    SELECT re."City", re."RestaurantID", re."RestaurantName",
           SUM(o."FinalAmount") AS revenue,
           DENSE_RANK() OVER (PARTITION BY re."City" ORDER BY SUM(o."FinalAmount") DESC) AS city_rank
    FROM orders o JOIN restaurants re ON re."RestaurantID"=o."RestaurantID"
    WHERE o."OrderStatus" IN ('Delivered','Delivered Late')
    GROUP BY 1,2,3
)
SELECT * FROM r WHERE city_rank <= 3 ORDER BY "City", city_rank;

-- 6. Restaurant scorecard including a composite score
SELECT * FROM vw_restaurant_scorecard ORDER BY composite_performance_score DESC NULLS LAST;

-- 7. Cuisine cancellation and late-delivery rates
SELECT r."Cuisine",
       COUNT(*) AS total_orders,
       ROUND(100.0*COUNT(*) FILTER (WHERE o."OrderStatus"='Cancelled')/COUNT(*),2) AS cancellation_rate_pct,
       ROUND(100.0*COUNT(*) FILTER (WHERE o."OrderStatus"='Delivered Late')/
          NULLIF(COUNT(*) FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),0),2) AS late_rate_pct
FROM orders o JOIN restaurants r ON r."RestaurantID"=o."RestaurantID"
GROUP BY r."Cuisine" ORDER BY late_rate_pct DESC NULLS LAST;

-- 8. Customer value summary
SELECT c."CustomerID", c."Name", c."Membership",
       COUNT(DISTINCT o."OrderID") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')) AS completed_orders,
       COALESCE(SUM(o."FinalAmount") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),0) AS lifetime_spend,
       MAX(o."OrderDate") AS last_order_date
FROM customers c LEFT JOIN orders o ON o."CustomerID"=c."CustomerID"
GROUP BY 1,2,3 ORDER BY lifetime_spend DESC;

-- 9. True month-over-month customer spend (not consecutive-order deltas)
WITH cm AS (
    SELECT "CustomerID", DATE_TRUNC('month',"OrderDate") AS month, SUM("FinalAmount") AS spend
    FROM orders WHERE "OrderStatus" IN ('Delivered','Delivered Late')
    GROUP BY 1,2
)
SELECT *, LAG(spend) OVER (PARTITION BY "CustomerID" ORDER BY month) AS prior_month_spend,
       spend - LAG(spend) OVER (PARTITION BY "CustomerID" ORDER BY month) AS change_vs_prior_month
FROM cm;

-- 10. Promotion effectiveness
SELECT p."CampaignName", p."CouponCode", p."DiscountPercentage",
       COUNT(o."OrderID") AS uses,
       ROUND(SUM(o."Discount"),2) AS discount_cost,
       ROUND(SUM(o."FinalAmount") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),2) AS completed_revenue,
       ROUND(AVG(o."FinalAmount") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),2) AS avg_completed_order_value
FROM promotions p LEFT JOIN orders o ON o."CouponCode"=p."CouponCode"
GROUP BY 1,2,3 ORDER BY completed_revenue DESC NULLS LAST;

-- 11. Payment success by method
SELECT "PaymentMethod", COUNT(*) AS payments,
       ROUND(100.0*COUNT(*) FILTER (WHERE "PaymentStatus"='Success')/COUNT(*),2) AS success_rate_pct
FROM payments GROUP BY "PaymentMethod" ORDER BY success_rate_pct DESC;

-- 12. Delivery-partner performance based on completed orders only
SELECT dp."DeliveryPartnerID", dp."Name", dp."VehicleType", dp."City",
       COUNT(o."OrderID") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')) AS completed_orders,
       ROUND(AVG(o."DeliveryTimeMinutes") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),2) AS avg_delivery_minutes,
       ROUND(100.0*COUNT(o."OrderID") FILTER (WHERE o."OrderStatus"='Delivered Late') /
          NULLIF(COUNT(o."OrderID") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),0),2) AS late_rate_pct
FROM delivery_partners dp LEFT JOIN orders o ON o."DeliveryPartnerID"=dp."DeliveryPartnerID"
GROUP BY 1,2,3,4 ORDER BY completed_orders DESC;

-- 13. Traffic matched to order hour (coarse SQL alternative; Python pipeline uses nearest timestamp)
SELECT t."TrafficLevel",
       ROUND(AVG(o."DeliveryTimeMinutes"),2) AS avg_delivery_minutes,
       COUNT(*) AS matched_orders
FROM orders o
JOIN restaurants r ON r."RestaurantID"=o."RestaurantID"
JOIN traffic t ON t."City"=r."City" AND t."Date"=o."OrderDate"
              AND EXTRACT(HOUR FROM t."Time")=EXTRACT(HOUR FROM o."OrderTime")
WHERE o."OrderStatus" IN ('Delivered','Delivered Late')
GROUP BY t."TrafficLevel" ORDER BY avg_delivery_minutes DESC;

-- 14. Weather effect
SELECT w."WeatherCondition",
       ROUND(AVG(o."DeliveryTimeMinutes"),2) AS avg_delivery_minutes,
       ROUND(AVG(w."Rainfall"),2) AS avg_rainfall,
       COUNT(*) AS orders
FROM orders o
JOIN restaurants r ON r."RestaurantID"=o."RestaurantID"
JOIN weather w ON w."City"=r."City" AND w."Date"=o."OrderDate"
WHERE o."OrderStatus" IN ('Delivered','Delivered Late')
GROUP BY w."WeatherCondition" ORDER BY avg_delivery_minutes DESC;

-- 15. Explicit source-data structural audit
SELECT
    ROUND(100.0*AVG((customer_city <> restaurant_city)::int),2) AS customer_restaurant_mismatch_pct,
    ROUND(100.0*AVG((restaurant_city <> partner_city)::int),2) AS restaurant_partner_mismatch_pct
FROM vw_order_base;
