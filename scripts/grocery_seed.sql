-- Grocery store sample schema + seed data for DataPoint-driven testing.

DROP TABLE IF EXISTS public.grocery_waste_events CASCADE;
DROP TABLE IF EXISTS public.grocery_purchase_orders CASCADE;
DROP TABLE IF EXISTS public.grocery_sales_transactions CASCADE;
DROP TABLE IF EXISTS public.grocery_inventory_snapshots CASCADE;
DROP TABLE IF EXISTS public.grocery_products CASCADE;
DROP TABLE IF EXISTS public.grocery_suppliers CASCADE;
DROP TABLE IF EXISTS public.grocery_stores CASCADE;

CREATE TABLE public.grocery_stores (
    store_id SERIAL PRIMARY KEY,
    store_code TEXT NOT NULL UNIQUE,
    store_name TEXT NOT NULL,
    city TEXT NOT NULL,
    region TEXT NOT NULL,
    opened_at DATE NOT NULL
);

CREATE TABLE public.grocery_suppliers (
    supplier_id SERIAL PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL,
    contact_email TEXT NOT NULL
);

CREATE TABLE public.grocery_products (
    product_id SERIAL PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_cost NUMERIC(10,2) NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL,
    is_perishable BOOLEAN NOT NULL DEFAULT true,
    reorder_level INTEGER NOT NULL,
    supplier_id INTEGER NOT NULL REFERENCES public.grocery_suppliers(supplier_id)
);

CREATE TABLE public.grocery_inventory_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    store_id INTEGER NOT NULL REFERENCES public.grocery_stores(store_id),
    product_id INTEGER NOT NULL REFERENCES public.grocery_products(product_id),
    on_hand_qty INTEGER NOT NULL,
    reserved_qty INTEGER NOT NULL DEFAULT 0,
    UNIQUE(snapshot_date, store_id, product_id)
);

CREATE TABLE public.grocery_sales_transactions (
    txn_id SERIAL PRIMARY KEY,
    sold_at TIMESTAMP NOT NULL,
    business_date DATE NOT NULL,
    store_id INTEGER NOT NULL REFERENCES public.grocery_stores(store_id),
    product_id INTEGER NOT NULL REFERENCES public.grocery_products(product_id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL,
    discount_amount NUMERIC(10,2) NOT NULL DEFAULT 0,
    total_amount NUMERIC(12,2) NOT NULL
);

CREATE TABLE public.grocery_purchase_orders (
    po_id SERIAL PRIMARY KEY,
    ordered_at TIMESTAMP NOT NULL,
    expected_at TIMESTAMP NOT NULL,
    received_at TIMESTAMP,
    supplier_id INTEGER NOT NULL REFERENCES public.grocery_suppliers(supplier_id),
    store_id INTEGER NOT NULL REFERENCES public.grocery_stores(store_id),
    product_id INTEGER NOT NULL REFERENCES public.grocery_products(product_id),
    ordered_qty INTEGER NOT NULL,
    received_qty INTEGER,
    unit_cost NUMERIC(10,2) NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE public.grocery_waste_events (
    waste_id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    store_id INTEGER NOT NULL REFERENCES public.grocery_stores(store_id),
    product_id INTEGER NOT NULL REFERENCES public.grocery_products(product_id),
    quantity INTEGER NOT NULL,
    reason TEXT NOT NULL,
    estimated_cost NUMERIC(12,2) NOT NULL
);

INSERT INTO public.grocery_stores (store_code, store_name, city, region, opened_at) VALUES
('ST001', 'Downtown Fresh', 'Austin', 'South', '2020-03-01'),
('ST002', 'Midtown Market', 'Austin', 'South', '2021-07-15'),
('ST003', 'Lakeside Grocers', 'Dallas', 'North', '2019-11-20');

INSERT INTO public.grocery_suppliers (supplier_name, lead_time_days, contact_email) VALUES
('FarmLine Produce', 2, 'ops@farmline.example.com'),
('Texas Dairy Co', 1, 'supply@texasdairy.example.com'),
('Pantry Wholesale', 4, 'logistics@pantrywholesale.example.com');

INSERT INTO public.grocery_products (
    sku, product_name, category, unit_cost, unit_price, is_perishable, reorder_level, supplier_id
) VALUES
('APL-01', 'Apple Gala 1lb', 'produce', 1.20, 2.49, true, 40, 1),
('BAN-01', 'Banana Bunch', 'produce', 0.90, 1.99, true, 50, 1),
('MLK-01', 'Whole Milk 1L', 'dairy', 1.05, 2.39, true, 60, 2),
('EGG-12', 'Eggs 12ct', 'dairy', 1.80, 3.79, true, 45, 2),
('BRD-01', 'Wheat Bread', 'bakery', 1.10, 2.99, true, 35, 3),
('PST-01', 'Pasta 500g', 'pantry', 0.70, 1.89, false, 80, 3),
('RCE-01', 'Rice 1kg', 'pantry', 1.40, 3.49, false, 70, 3),
('OIL-01', 'Olive Oil 500ml', 'pantry', 3.20, 6.99, false, 25, 3);

INSERT INTO public.grocery_inventory_snapshots
(snapshot_date, store_id, product_id, on_hand_qty, reserved_qty)
VALUES
('2026-02-01', 1, 1, 120, 4),
('2026-02-01', 1, 2, 98, 2),
('2026-02-01', 1, 3, 140, 3),
('2026-02-01', 1, 4, 72, 1),
('2026-02-01', 1, 5, 54, 0),
('2026-02-01', 2, 1, 88, 2),
('2026-02-01', 2, 3, 118, 2),
('2026-02-01', 2, 6, 165, 5),
('2026-02-01', 2, 7, 130, 4),
('2026-02-01', 3, 2, 110, 1),
('2026-02-01', 3, 4, 84, 2),
('2026-02-01', 3, 8, 46, 1),
('2026-02-02', 1, 1, 102, 3),
('2026-02-02', 1, 3, 126, 2),
('2026-02-02', 2, 6, 149, 3),
('2026-02-02', 3, 8, 41, 1);

INSERT INTO public.grocery_sales_transactions
(sold_at, business_date, store_id, product_id, quantity, unit_price, discount_amount, total_amount)
VALUES
('2026-02-01 08:12:00', '2026-02-01', 1, 1, 4, 2.49, 0.50, 9.46),
('2026-02-01 09:43:00', '2026-02-01', 1, 3, 3, 2.39, 0.00, 7.17),
('2026-02-01 10:08:00', '2026-02-01', 1, 5, 2, 2.99, 0.00, 5.98),
('2026-02-01 11:21:00', '2026-02-01', 2, 6, 6, 1.89, 1.00, 10.34),
('2026-02-01 12:04:00', '2026-02-01', 2, 7, 5, 3.49, 0.00, 17.45),
('2026-02-01 13:47:00', '2026-02-01', 3, 2, 7, 1.99, 0.80, 13.13),
('2026-02-01 15:15:00', '2026-02-01', 3, 8, 2, 6.99, 0.00, 13.98),
('2026-02-01 17:32:00', '2026-02-01', 3, 4, 3, 3.79, 0.00, 11.37),
('2026-02-02 08:05:00', '2026-02-02', 1, 1, 5, 2.49, 0.00, 12.45),
('2026-02-02 09:16:00', '2026-02-02', 1, 3, 4, 2.39, 0.00, 9.56),
('2026-02-02 11:03:00', '2026-02-02', 1, 4, 3, 3.79, 0.40, 10.97),
('2026-02-02 12:42:00', '2026-02-02', 2, 6, 8, 1.89, 1.20, 13.92),
('2026-02-02 14:20:00', '2026-02-02', 2, 7, 4, 3.49, 0.00, 13.96),
('2026-02-02 16:55:00', '2026-02-02', 3, 2, 6, 1.99, 0.50, 11.44),
('2026-02-02 18:17:00', '2026-02-02', 3, 8, 1, 6.99, 0.00, 6.99),
('2026-02-03 08:34:00', '2026-02-03', 1, 5, 4, 2.99, 0.00, 11.96),
('2026-02-03 10:01:00', '2026-02-03', 1, 3, 5, 2.39, 0.60, 11.35),
('2026-02-03 12:18:00', '2026-02-03', 2, 1, 3, 2.49, 0.00, 7.47),
('2026-02-03 13:44:00', '2026-02-03', 2, 6, 7, 1.89, 0.70, 12.53),
('2026-02-03 15:26:00', '2026-02-03', 3, 4, 4, 3.79, 0.50, 14.66),
('2026-02-03 17:09:00', '2026-02-03', 3, 2, 5, 1.99, 0.30, 9.65),
('2026-02-04 09:05:00', '2026-02-04', 1, 7, 4, 3.49, 0.00, 13.96),
('2026-02-04 11:11:00', '2026-02-04', 2, 3, 6, 2.39, 0.90, 13.44),
('2026-02-04 16:30:00', '2026-02-04', 3, 8, 2, 6.99, 0.00, 13.98);

INSERT INTO public.grocery_purchase_orders
(ordered_at, expected_at, received_at, supplier_id, store_id, product_id, ordered_qty, received_qty, unit_cost, status)
VALUES
('2026-01-29 06:00:00', '2026-01-31 10:00:00', '2026-01-31 09:20:00', 1, 1, 1, 80, 80, 1.20, 'received'),
('2026-01-30 06:15:00', '2026-01-31 09:00:00', '2026-01-31 08:45:00', 2, 1, 3, 90, 90, 1.05, 'received'),
('2026-01-31 07:00:00', '2026-02-04 11:00:00', NULL, 3, 2, 6, 140, NULL, 0.70, 'in_transit'),
('2026-02-01 07:30:00', '2026-02-03 10:00:00', '2026-02-03 10:40:00', 1, 3, 2, 120, 116, 0.90, 'partial'),
('2026-02-01 08:10:00', '2026-02-02 10:00:00', '2026-02-02 09:55:00', 2, 3, 4, 60, 60, 1.80, 'received'),
('2026-02-02 06:55:00', '2026-02-06 14:00:00', NULL, 3, 1, 8, 40, NULL, 3.20, 'in_transit'),
('2026-02-02 07:20:00', '2026-02-03 08:00:00', '2026-02-03 08:12:00', 2, 2, 3, 75, 75, 1.05, 'received'),
('2026-02-03 06:40:00', '2026-02-07 12:00:00', NULL, 3, 3, 7, 110, NULL, 1.40, 'submitted');

INSERT INTO public.grocery_waste_events
(event_date, store_id, product_id, quantity, reason, estimated_cost)
VALUES
('2026-02-01', 1, 1, 3, 'damaged shipment', 3.60),
('2026-02-01', 2, 3, 4, 'expired', 4.20),
('2026-02-02', 3, 2, 5, 'overripe', 4.50),
('2026-02-03', 1, 5, 2, 'stale', 2.20),
('2026-02-03', 3, 4, 3, 'broken pack', 5.40),
('2026-02-04', 2, 1, 2, 'quality reject', 2.40);
