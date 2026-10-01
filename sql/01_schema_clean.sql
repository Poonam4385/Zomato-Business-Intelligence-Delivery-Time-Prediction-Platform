-- Clean constrained PostgreSQL schema for the Zomato BI + ML project.
-- No sample INSERT statements are included, so bulk loading cannot collide with demo keys.

DROP VIEW IF EXISTS vw_customer_rfm CASCADE;
DROP VIEW IF EXISTS vw_restaurant_scorecard CASCADE;
DROP VIEW IF EXISTS vw_order_base CASCADE;
DROP TABLE IF EXISTS customer_feedback CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS menu CASCADE;
DROP TABLE IF EXISTS promotions CASCADE;
DROP TABLE IF EXISTS delivery_partners CASCADE;
DROP TABLE IF EXISTS restaurants CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS traffic CASCADE;
DROP TABLE IF EXISTS weather CASCADE;
DROP TABLE IF EXISTS cities CASCADE;

CREATE TABLE cities (
    "CityID" INTEGER PRIMARY KEY,
    "City" VARCHAR(100) NOT NULL UNIQUE,
    "Population" BIGINT CHECK ("Population" > 0),
    "Region" VARCHAR(50),
    "AverageIncome" NUMERIC(12,2)
);

CREATE TABLE customers (
    "CustomerID" INTEGER PRIMARY KEY,
    "Name" VARCHAR(150) NOT NULL,
    "Age" SMALLINT CHECK ("Age" BETWEEN 0 AND 100),
    "Gender" VARCHAR(20), "Phone" VARCHAR(30), "Email" VARCHAR(150),
    "City" VARCHAR(100) NOT NULL REFERENCES cities("City"),
    "State" VARCHAR(100), "Pincode" VARCHAR(20),
    "RegistrationDate" DATE,
    "Membership" VARCHAR(30),
    "TotalOrders" INTEGER DEFAULT 0 CHECK ("TotalOrders" >= 0),
    "PreferredCuisine" VARCHAR(50)
);

CREATE TABLE restaurants (
    "RestaurantID" INTEGER PRIMARY KEY,
    "RestaurantName" VARCHAR(150) NOT NULL,
    "Cuisine" VARCHAR(50),
    "City" VARCHAR(100) NOT NULL REFERENCES cities("City"),
    "Area" VARCHAR(100), "OpeningTime" TIME, "ClosingTime" TIME,
    "Rating" NUMERIC(3,1) CHECK ("Rating" BETWEEN 1 AND 5),
    "AverageCost" NUMERIC(10,2) CHECK ("AverageCost" >= 0),
    "OwnerName" VARCHAR(150), "RestaurantType" VARCHAR(50),
    "Latitude" NUMERIC(10,6), "Longitude" NUMERIC(10,6)
);

CREATE TABLE delivery_partners (
    "DeliveryPartnerID" INTEGER PRIMARY KEY,
    "Name" VARCHAR(150) NOT NULL,
    "Age" SMALLINT CHECK ("Age" BETWEEN 16 AND 70),
    "Gender" VARCHAR(20), "VehicleType" VARCHAR(30), "JoiningDate" DATE,
    "City" VARCHAR(100) NOT NULL REFERENCES cities("City"),
    "Rating" NUMERIC(3,1) CHECK ("Rating" BETWEEN 1 AND 5),
    "CompletedDeliveries" INTEGER CHECK ("CompletedDeliveries" >= 0),
    "AverageDeliveryTime" NUMERIC(8,2) CHECK ("AverageDeliveryTime" >= 0)
);

CREATE TABLE promotions (
    "PromotionID" INTEGER PRIMARY KEY,
    "CouponCode" VARCHAR(30) NOT NULL UNIQUE,
    "DiscountPercentage" NUMERIC(6,2) CHECK ("DiscountPercentage" BETWEEN 0 AND 100),
    "CampaignName" VARCHAR(150), "StartDate" DATE, "EndDate" DATE,
    CHECK ("EndDate" IS NULL OR "StartDate" IS NULL OR "EndDate" >= "StartDate")
);

CREATE TABLE menu (
    "FoodItemID" INTEGER PRIMARY KEY,
    "RestaurantID" INTEGER NOT NULL REFERENCES restaurants("RestaurantID"),
    "FoodName" VARCHAR(150) NOT NULL, "Category" VARCHAR(50),
    "Price" NUMERIC(10,2) CHECK ("Price" >= 0),
    "PreparationTime" SMALLINT CHECK ("PreparationTime" >= 0),
    "Calories" INTEGER, "Availability" VARCHAR(10)
);

CREATE TABLE orders (
    "OrderID" INTEGER PRIMARY KEY,
    "CustomerID" INTEGER NOT NULL REFERENCES customers("CustomerID"),
    "RestaurantID" INTEGER NOT NULL REFERENCES restaurants("RestaurantID"),
    "DeliveryPartnerID" INTEGER NOT NULL REFERENCES delivery_partners("DeliveryPartnerID"),
    "OrderDate" DATE NOT NULL, "OrderTime" TIME,
    "DeliveryTimeMinutes" SMALLINT CHECK ("DeliveryTimeMinutes" >= 0),
    "FoodCost" NUMERIC(12,2) CHECK ("FoodCost" >= 0),
    "DeliveryFee" NUMERIC(10,2) CHECK ("DeliveryFee" >= 0),
    "Discount" NUMERIC(10,2) CHECK ("Discount" >= 0),
    "CouponCode" VARCHAR(30) REFERENCES promotions("CouponCode"),
    "GST" NUMERIC(10,2) CHECK ("GST" >= 0),
    "FinalAmount" NUMERIC(12,2) CHECK ("FinalAmount" >= 0),
    "OrderStatus" VARCHAR(30) NOT NULL,
    "PaymentMethod" VARCHAR(30)
);

CREATE TABLE order_items (
    "OrderItemID" INTEGER PRIMARY KEY,
    "OrderID" INTEGER NOT NULL REFERENCES orders("OrderID"),
    "FoodItemID" INTEGER NOT NULL REFERENCES menu("FoodItemID"),
    "Quantity" SMALLINT NOT NULL CHECK ("Quantity" > 0),
    "UnitPrice" NUMERIC(10,2) CHECK ("UnitPrice" >= 0),
    "TotalPrice" NUMERIC(12,2) CHECK ("TotalPrice" >= 0)
);

CREATE TABLE payments (
    "PaymentID" INTEGER PRIMARY KEY,
    "OrderID" INTEGER NOT NULL REFERENCES orders("OrderID"),
    "PaymentMethod" VARCHAR(30), "PaymentStatus" VARCHAR(20),
    "TransactionID" VARCHAR(50), "PaymentDate" DATE
);

CREATE TABLE customer_feedback (
    "FeedbackID" INTEGER PRIMARY KEY,
    "OrderID" INTEGER NOT NULL REFERENCES orders("OrderID"),
    "CustomerRating" SMALLINT CHECK ("CustomerRating" BETWEEN 1 AND 5),
    "DeliveryRating" SMALLINT CHECK ("DeliveryRating" BETWEEN 1 AND 5),
    "FoodRating" SMALLINT CHECK ("FoodRating" BETWEEN 1 AND 5),
    "Review" TEXT, "Sentiment" VARCHAR(20)
);

CREATE TABLE weather (
    "WeatherID" INTEGER PRIMARY KEY,
    "City" VARCHAR(100) NOT NULL REFERENCES cities("City"),
    "Date" DATE NOT NULL,
    "Temperature" NUMERIC(6,2), "Rainfall" NUMERIC(8,2),
    "Humidity" NUMERIC(6,2) CHECK ("Humidity" BETWEEN 0 AND 100),
    "WeatherCondition" VARCHAR(30)
);

CREATE TABLE traffic (
    "TrafficID" INTEGER PRIMARY KEY,
    "City" VARCHAR(100) NOT NULL REFERENCES cities("City"),
    "Date" DATE NOT NULL, "Time" TIME,
    "TrafficLevel" VARCHAR(20), "AverageSpeed" NUMERIC(8,2)
);

CREATE INDEX idx_orders_date ON orders("OrderDate");
CREATE INDEX idx_orders_customer ON orders("CustomerID");
CREATE INDEX idx_orders_restaurant ON orders("RestaurantID");
CREATE INDEX idx_orders_partner ON orders("DeliveryPartnerID");
CREATE INDEX idx_orders_status ON orders("OrderStatus");
CREATE INDEX idx_order_items_order ON order_items("OrderID");
CREATE INDEX idx_feedback_order ON customer_feedback("OrderID");
CREATE INDEX idx_weather_city_date ON weather("City", "Date");
CREATE INDEX idx_traffic_city_date_time ON traffic("City", "Date", "Time");
