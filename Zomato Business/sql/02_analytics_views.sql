-- Corrected analytics layer. Completed orders include both Delivered and Delivered Late.

CREATE OR REPLACE VIEW vw_order_base AS
SELECT
    o."OrderID", o."CustomerID", o."RestaurantID", o."DeliveryPartnerID",
    o."OrderDate", o."OrderTime", o."DeliveryTimeMinutes", o."FoodCost",
    o."DeliveryFee", o."Discount", o."GST", o."FinalAmount", o."OrderStatus", o."PaymentMethod",
    c."City" AS customer_city, c."Membership", c."PreferredCuisine",
    r."RestaurantName", r."City" AS restaurant_city, r."Cuisine", r."Rating" AS restaurant_rating,
    dp."City" AS partner_city, dp."VehicleType", dp."Rating" AS partner_rating,
    CASE WHEN c."City" <> r."City" THEN 1 ELSE 0 END AS customer_restaurant_city_mismatch,
    CASE WHEN r."City" <> dp."City" THEN 1 ELSE 0 END AS restaurant_partner_city_mismatch,
    CASE WHEN o."OrderStatus" IN ('Delivered','Delivered Late') THEN 1 ELSE 0 END AS is_completed,
    CASE WHEN o."OrderStatus" = 'Delivered Late' THEN 1 ELSE 0 END AS is_late,
    CASE WHEN o."OrderStatus" = 'Cancelled' THEN 1 ELSE 0 END AS is_cancelled
FROM orders o
JOIN customers c ON c."CustomerID" = o."CustomerID"
JOIN restaurants r ON r."RestaurantID" = o."RestaurantID"
JOIN delivery_partners dp ON dp."DeliveryPartnerID" = o."DeliveryPartnerID";

CREATE OR REPLACE VIEW vw_restaurant_scorecard AS
WITH s AS (
    SELECT
        r."RestaurantID", r."RestaurantName", r."City", r."Cuisine", r."Rating",
        COUNT(o."OrderID") AS total_orders,
        COUNT(o."OrderID") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')) AS completed_orders,
        SUM(o."FinalAmount") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')) AS revenue,
        AVG(o."DeliveryTimeMinutes") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')) AS avg_delivery_minutes,
        100.0 * COUNT(o."OrderID") FILTER (WHERE o."OrderStatus"='Cancelled') / NULLIF(COUNT(o."OrderID"),0) AS cancellation_rate_pct,
        100.0 * COUNT(o."OrderID") FILTER (WHERE o."OrderStatus"='Delivered Late') /
            NULLIF(COUNT(o."OrderID") FILTER (WHERE o."OrderStatus" IN ('Delivered','Delivered Late')),0) AS late_rate_pct
    FROM restaurants r
    LEFT JOIN orders o ON o."RestaurantID" = r."RestaurantID"
    GROUP BY 1,2,3,4,5
)
SELECT *,
    ROUND((
        COALESCE("Rating",3.0)/5.0 * 35 +
        (1 - LEAST(COALESCE(cancellation_rate_pct,0),100)/100.0) * 20 +
        (1 - LEAST(COALESCE(late_rate_pct,0),100)/100.0) * 20 +
        LEAST(COALESCE(completed_orders,0)/50.0,1.0) * 10 +
        LEAST(COALESCE(revenue,0)/50000.0,1.0) * 15
    )::numeric, 2) AS composite_performance_score
FROM s;

CREATE OR REPLACE VIEW vw_customer_rfm AS
WITH snapshot AS (SELECT MAX("OrderDate") AS max_date FROM orders),
completed AS (
    SELECT * FROM orders WHERE "OrderStatus" IN ('Delivered','Delivered Late')
), rfm AS (
    SELECT
        c."CustomerID", c."Name", c."City", c."Membership",
        MAX(o."OrderDate") AS last_order_date,
        COUNT(DISTINCT o."OrderID") AS frequency,
        COALESCE(SUM(o."FinalAmount"),0) AS monetary
    FROM customers c
    LEFT JOIN completed o ON o."CustomerID" = c."CustomerID"
    GROUP BY 1,2,3,4
)
SELECT rfm.*,
       (snapshot.max_date - rfm.last_order_date) AS recency_days
FROM rfm CROSS JOIN snapshot;
