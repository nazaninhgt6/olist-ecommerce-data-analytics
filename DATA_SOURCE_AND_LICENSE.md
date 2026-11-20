# Data provenance, definitions and licensing

- **Source:** Olist, *Brazilian E-Commerce Public Dataset*, https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Period:** historical transactions from 2016–2018. The filtered delivered-order monthly report spans 2016-09 through 2018-08.
- **Dataset license:** CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
- **Attribution:** Data supplied by Olist via Kaggle. This repository analyzes and visualizes those data and is not affiliated with or endorsed by Olist.
- **Distribution choice:** The original raw CSVs are not redistributed. `reports/` and the interactive dashboard contain only analytical aggregates derived from the dataset; their reuse is subject to applicable source-license terms (attribution, non-commercial, share alike).
- **Code:** Source code is supplied for reproducibility and review; no separate open-source license is expressly granted here. The repository owner may choose a code license separately.

## Important definitions

- `delivered_item_value_brl` is the sum of `order_items.price` for delivered orders with item rows; freight is excluded. It does **not** represent profit or cash collected.
- `delivered_orders_with_items` counts distinct delivered order IDs that have item records.
- `repeat_customer_percentage` counts distinct `customer_unique_id` with two or more delivered orders divided by all distinct customers in scope; it is not cohort retention.
- `late_delivery_percentage` compares actual and estimated delivery **calendar dates** over records with valid purchase, actual-delivery and estimated-delivery dates and nonnegative delivery duration.
- Geography is measured by `customer_state` (customer's state), not seller location.
- Monthly metrics use order **purchase month**, not delivery month; sparse early periods must not be interpreted as mature sales performance.
- One delivered order lacks a linked payment record; the analysis uses item prices rather than paid revenue.

## Caution on interpretation

Findings and recommendations are exploratory hypotheses. The data do not establish causality, lifetime customer loyalty, or the operational performance of the business today.
