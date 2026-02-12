-- Fintech bank sample schema + seed data for DataPoint-driven testing.

DROP TABLE IF EXISTS public.bank_loan_payments CASCADE;
DROP TABLE IF EXISTS public.bank_cards CASCADE;
DROP TABLE IF EXISTS public.bank_transactions CASCADE;
DROP TABLE IF EXISTS public.bank_loans CASCADE;
DROP TABLE IF EXISTS public.bank_accounts CASCADE;
DROP TABLE IF EXISTS public.bank_fx_rates CASCADE;
DROP TABLE IF EXISTS public.bank_customers CASCADE;

CREATE TABLE public.bank_customers (
    customer_id SERIAL PRIMARY KEY,
    customer_code TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    country TEXT NOT NULL,
    segment TEXT NOT NULL,
    kyc_status TEXT NOT NULL,
    created_at DATE NOT NULL
);

CREATE TABLE public.bank_accounts (
    account_id SERIAL PRIMARY KEY,
    account_number TEXT NOT NULL UNIQUE,
    customer_id INTEGER NOT NULL REFERENCES public.bank_customers(customer_id),
    account_type TEXT NOT NULL,
    currency_code TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at DATE NOT NULL,
    current_balance NUMERIC(14,2) NOT NULL
);

CREATE TABLE public.bank_transactions (
    txn_id SERIAL PRIMARY KEY,
    posted_at TIMESTAMP NOT NULL,
    business_date DATE NOT NULL,
    account_id INTEGER NOT NULL REFERENCES public.bank_accounts(account_id),
    counterparty_account TEXT,
    txn_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    amount NUMERIC(14,2) NOT NULL,
    fee_amount NUMERIC(10,2) NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    reference_text TEXT
);

CREATE TABLE public.bank_cards (
    card_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES public.bank_customers(customer_id),
    account_id INTEGER NOT NULL REFERENCES public.bank_accounts(account_id),
    card_type TEXT NOT NULL,
    network TEXT NOT NULL,
    status TEXT NOT NULL,
    issued_at DATE NOT NULL,
    blocked_at DATE
);

CREATE TABLE public.bank_loans (
    loan_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES public.bank_customers(customer_id),
    repayment_account_id INTEGER REFERENCES public.bank_accounts(account_id),
    loan_type TEXT NOT NULL,
    principal_amount NUMERIC(14,2) NOT NULL,
    interest_rate NUMERIC(5,4) NOT NULL,
    disbursed_at DATE NOT NULL,
    maturity_date DATE NOT NULL,
    status TEXT NOT NULL,
    days_past_due INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE public.bank_loan_payments (
    payment_id SERIAL PRIMARY KEY,
    loan_id INTEGER NOT NULL REFERENCES public.bank_loans(loan_id),
    payment_date DATE NOT NULL,
    amount NUMERIC(14,2) NOT NULL,
    principal_component NUMERIC(14,2) NOT NULL,
    interest_component NUMERIC(14,2) NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE public.bank_fx_rates (
    rate_date DATE NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    rate NUMERIC(12,6) NOT NULL,
    PRIMARY KEY (rate_date, base_currency, quote_currency)
);

INSERT INTO public.bank_customers
(customer_code, full_name, email, country, segment, kyc_status, created_at)
VALUES
('CUST001', 'Ada Okafor', 'ada.okafor@example.com', 'NG', 'retail', 'verified', '2023-05-10'),
('CUST002', 'Noah Mensah', 'noah.mensah@example.com', 'GH', 'retail', 'verified', '2022-11-18'),
('CUST003', 'Luna Ibrahim', 'luna.ibrahim@example.com', 'NG', 'sme', 'verified', '2021-08-02'),
('CUST004', 'Kai Daniels', 'kai.daniels@example.com', 'US', 'sme', 'pending_review', '2024-01-22');

INSERT INTO public.bank_accounts
(account_number, customer_id, account_type, currency_code, status, opened_at, current_balance)
VALUES
('0010000011', 1, 'checking', 'USD', 'active', '2023-05-12', 12540.75),
('0010000012', 1, 'savings', 'USD', 'active', '2023-05-13', 40210.10),
('0010000021', 2, 'checking', 'USD', 'active', '2022-11-20', 8090.00),
('0010000031', 3, 'checking', 'USD', 'active', '2021-08-03', 23050.45),
('0010000032', 3, 'savings', 'EUR', 'active', '2021-09-01', 15500.00),
('0010000041', 4, 'checking', 'USD', 'restricted', '2024-01-23', 2200.00);

INSERT INTO public.bank_transactions
(posted_at, business_date, account_id, counterparty_account, txn_type, direction, amount, fee_amount, status, reference_text)
VALUES
('2026-02-01 08:14:00', '2026-02-01', 1, 'EXT-9001', 'card_purchase', 'debit', 64.20, 0.35, 'posted', 'Groceries'),
('2026-02-01 09:22:00', '2026-02-01', 1, 'EXT-3110', 'transfer', 'credit', 500.00, 0.00, 'posted', 'Inbound transfer'),
('2026-02-01 11:05:00', '2026-02-01', 2, 'EXT-4102', 'interest_credit', 'credit', 35.12, 0.00, 'posted', 'Monthly interest'),
('2026-02-01 12:47:00', '2026-02-01', 3, 'EXT-9920', 'bill_payment', 'debit', 120.00, 0.50, 'posted', 'Utility bill'),
('2026-02-01 13:09:00', '2026-02-01', 4, 'EXT-1242', 'transfer', 'debit', 1750.00, 1.25, 'posted', 'Vendor payment'),
('2026-02-01 14:42:00', '2026-02-01', 5, 'EXT-5570', 'fx_transfer', 'debit', 420.00, 2.40, 'declined', 'Insufficient funds'),
('2026-02-02 09:17:00', '2026-02-02', 1, 'EXT-9012', 'card_purchase', 'debit', 42.85, 0.20, 'posted', 'Fuel'),
('2026-02-02 10:24:00', '2026-02-02', 2, 'EXT-4300', 'transfer', 'debit', 1200.00, 0.75, 'posted', 'Investment transfer'),
('2026-02-02 11:58:00', '2026-02-02', 3, 'EXT-8110', 'cash_withdrawal', 'debit', 300.00, 1.00, 'posted', 'ATM withdrawal'),
('2026-02-02 12:11:00', '2026-02-02', 4, 'EXT-1553', 'transfer', 'credit', 2200.00, 0.00, 'posted', 'Client receipt'),
('2026-02-02 13:54:00', '2026-02-02', 6, 'EXT-1001', 'card_purchase', 'debit', 75.00, 0.30, 'reversed', 'Card blocked'),
('2026-02-02 16:44:00', '2026-02-02', 5, 'EXT-6611', 'fx_transfer', 'credit', 380.00, 0.00, 'posted', 'FX settlement'),
('2026-02-03 08:41:00', '2026-02-03', 1, 'EXT-7810', 'salary_credit', 'credit', 2400.00, 0.00, 'posted', 'Payroll'),
('2026-02-03 09:35:00', '2026-02-03', 3, 'EXT-1780', 'card_purchase', 'debit', 28.10, 0.15, 'posted', 'Coffee shop'),
('2026-02-03 10:50:00', '2026-02-03', 4, 'EXT-9911', 'transfer', 'debit', 980.00, 0.80, 'posted', 'Supplier payment'),
('2026-02-03 13:13:00', '2026-02-03', 2, 'EXT-4392', 'bill_payment', 'debit', 215.00, 0.50, 'posted', 'Insurance premium'),
('2026-02-03 14:07:00', '2026-02-03', 5, 'EXT-8821', 'fx_transfer', 'debit', 250.00, 1.40, 'posted', 'International vendor'),
('2026-02-03 17:21:00', '2026-02-03', 6, 'EXT-7312', 'card_purchase', 'debit', 49.90, 0.20, 'declined', 'Merchant timeout');

INSERT INTO public.bank_cards
(customer_id, account_id, card_type, network, status, issued_at, blocked_at)
VALUES
(1, 1, 'debit', 'visa', 'active', '2023-05-20', NULL),
(2, 3, 'debit', 'mastercard', 'active', '2022-12-02', NULL),
(3, 4, 'credit', 'visa', 'active', '2021-08-10', NULL),
(4, 6, 'debit', 'visa', 'blocked', '2024-01-25', '2026-01-30');

INSERT INTO public.bank_loans
(customer_id, repayment_account_id, loan_type, principal_amount, interest_rate, disbursed_at, maturity_date, status, days_past_due)
VALUES
(1, 1, 'personal', 15000.00, 0.0890, '2024-03-01', '2027-03-01', 'active', 0),
(2, 3, 'auto', 22000.00, 0.0725, '2023-09-15', '2028-09-15', 'active', 14),
(3, 4, 'working_capital', 85000.00, 0.1140, '2022-06-20', '2027-06-20', 'delinquent', 95),
(3, 5, 'equipment', 42000.00, 0.0980, '2025-01-11', '2029-01-11', 'active', 0);

INSERT INTO public.bank_loan_payments
(loan_id, payment_date, amount, principal_component, interest_component, status)
VALUES
(1, '2026-01-05', 620.00, 500.00, 120.00, 'posted'),
(1, '2026-02-05', 620.00, 504.00, 116.00, 'posted'),
(2, '2026-01-20', 740.00, 610.00, 130.00, 'posted'),
(2, '2026-02-20', 740.00, 615.00, 125.00, 'posted'),
(3, '2026-01-28', 0.00, 0.00, 0.00, 'missed'),
(4, '2026-02-09', 1100.00, 860.00, 240.00, 'posted');

INSERT INTO public.bank_fx_rates
(rate_date, base_currency, quote_currency, rate)
VALUES
('2026-02-01', 'USD', 'NGN', 1512.500000),
('2026-02-01', 'USD', 'EUR', 0.923000),
('2026-02-02', 'USD', 'NGN', 1518.750000),
('2026-02-02', 'USD', 'EUR', 0.926500),
('2026-02-03', 'USD', 'NGN', 1521.100000),
('2026-02-03', 'USD', 'EUR', 0.929200);
