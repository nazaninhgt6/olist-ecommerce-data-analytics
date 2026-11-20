#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Olist end-to-end data analytics portfolio.

1. Download/load the Olist Brazilian E-Commerce dataset.
2. Validate keys and relationships, create order-level data (no double counting).
3. Produce category, state, customer, delivery, and monthly performance reports.
4. Validate totals and build a fully offline, Persian-language HTML dashboard.

Requirements: pip install pandas plotly kagglehub
Run from scratch: python Olist_Full_Portfolio_Analysis.py
Run with local dataset: python Olist_Full_Portfolio_Analysis.py --data-dir /path/to/olist
Rebuild from existing reports: python Olist_Full_Portfolio_Analysis.py --reports-dir reports
Output directory: ./olist_output (change using --output-dir)

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Figures are item value (sum of price), excluding freight; they are NOT profit.
"""

# %% 01 — Configuration and imports
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import warnings

import pandas as pd
from plotly.offline import get_plotlyjs

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"
REQUIRED_REPORTS = (
    "category_performance.csv", "state_performance.csv",
    "customer_order_frequency.csv", "delivery_by_state.csv",
    "monthly_performance.csv", "metrics_snapshot.json",
)


def get_dataset_directory(data_dir: str | Path | None = None) -> Path:
    if data_dir:
        folder = Path(data_dir).expanduser().resolve()
    else:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError("Install kagglehub: pip install kagglehub") from exc
        print("Downloading/locating Olist dataset using kagglehub ...")
        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET)).resolve()
    if not (folder / "olist_orders_dataset.csv").is_file():
        raise FileNotFoundError(
            f"Olist CSV files were not found in {folder}. "
            "Pass the directory containing olist_orders_dataset.csv."
        )
    print("Dataset directory:", folder)
    return folder


def read_table(folder: Path, filename: str, required: set[str]) -> pd.DataFrame:
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {filename}: {sorted(missing)}")
    print(f"  {filename}: {len(df):,} rows")
    return df


def read_raw_tables(folder: Path) -> dict[str, pd.DataFrame]:
    print("Reading source tables ...")
    specs = {
        "orders": ("olist_orders_dataset.csv", {"order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"}),
        "customers": ("olist_customers_dataset.csv", {"customer_id", "customer_unique_id", "customer_state"}),
        "products": ("olist_products_dataset.csv", {"product_id", "product_category_name"}),
        "order_items": ("olist_order_items_dataset.csv", {"order_id", "order_item_id", "product_id", "price", "freight_value"}),
        "payments": ("olist_order_payments_dataset.csv", {"order_id", "payment_sequential", "payment_value"}),
        "translations": ("product_category_name_translation.csv", {"product_category_name", "product_category_name_english"}),
    }
    return {name: read_table(folder, filename, needed) for name, (filename, needed) in specs.items()}


# %% 02 — Data-quality checks, order-level table

def validate_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = [
        ("orders", ["order_id"]),
        ("customers", ["customer_id"]),
        ("products", ["product_id"]),
        ("order_items", ["order_id", "order_item_id"]),
        ("payments", ["order_id", "payment_sequential"]),
        ("translations", ["product_category_name"]),
    ]
    primary_rows = []
    for name, keys in checks:
        df = tables[name]
        missing = int(df[keys].isna().any(axis=1).sum())
        duplicates = int(df.duplicated(subset=keys).sum())
        primary_rows.append({"table": name, "rows": len(df), "key_columns": ",".join(keys),
                             "missing_keys": missing, "duplicate_keys": duplicates})
    primary_report = pd.DataFrame(primary_rows)
    if ((primary_report["missing_keys"] > 0) | (primary_report["duplicate_keys"] > 0)).any():
        raise ValueError("Duplicate or missing primary keys detected. Inspect key_validation.csv.")

    relations = [
        ("orders", "customer_id", "customers", "customer_id"),
        ("order_items", "order_id", "orders", "order_id"),
        ("order_items", "product_id", "products", "product_id"),
        ("payments", "order_id", "orders", "order_id"),
    ]
    relation_rows = []
    for child, child_key, parent, parent_key in relations:
        source = tables[child][child_key]
        missing = int(source.isna().sum())
        unmatched = int((source.notna() & ~source.isin(tables[parent][parent_key])).sum())
        relation_rows.append({"relation": f"{child}.{child_key} -> {parent}.{parent_key}",
                              "rows": len(source), "missing_keys": missing, "unmatched_keys": unmatched,
                              "valid_percentage": round(100 * (len(source)-missing-unmatched) / len(source), 2) if len(source) else 100})
    relation_report = pd.DataFrame(relation_rows)
    if (relation_report["unmatched_keys"] > 0).any():
        warnings.warn("Some foreign-key relationships are unmatched; examine relation_validation.csv")
    return primary_report, relation_report


def make_order_level(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"]
    items = tables["order_items"]
    payments = tables["payments"]
    item_summary = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        product_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    )
    payment_summary = payments.groupby("order_id", as_index=False).agg(
        payment_count=("payment_sequential", "count"),
        payment_value=("payment_value", "sum"),
    )
    customers = tables["customers"]
    cols = [c for c in ["customer_id", "customer_unique_id", "customer_city", "customer_state"] if c in customers]
    order_level = (orders.merge(customers[cols], on="customer_id", how="left", validate="many_to_one")
                   .merge(item_summary, on="order_id", how="left", validate="one_to_one")
                   .merge(payment_summary, on="order_id", how="left", validate="one_to_one"))
    assert len(order_level) == len(orders)
    assert order_level["order_id"].is_unique
    return order_level


def make_status_report(order_level: pd.DataFrame) -> pd.DataFrame:
    d = order_level.assign(missing_items=order_level["item_count"].isna(),
                           missing_payment=order_level["payment_count"].isna())
    result = d.groupby("order_status", as_index=False).agg(
        total_orders=("order_id", "size"), missing_items=("missing_items", "sum"),
        missing_payments=("missing_payment", "sum"))
    result["percentage"] = (result["total_orders"] / len(d) * 100)
    return result.sort_values("total_orders", ascending=False).reset_index(drop=True)


# %% 03 — Business analyses (same definitions as the user's notebook)

def analyze_olist(tables: dict[str, pd.DataFrame], order_level: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    sales = order_level.loc[(order_level["order_status"] == "delivered") &
                            order_level["product_value"].notna()].copy()
    if sales.empty:
        raise ValueError("No delivered orders with item records; unable to build dashboard.")
    expected = float(sales["product_value"].sum())
    delivered_ids = sales[["order_id"]].drop_duplicates()
    category_items = (tables["order_items"].merge(delivered_ids, on="order_id", how="inner", validate="many_to_one")
                      .merge(tables["products"][["product_id", "product_category_name"]], on="product_id", how="left", validate="many_to_one")
                      .merge(tables["translations"][["product_category_name", "product_category_name_english"]],
                             on="product_category_name", how="left", validate="many_to_one"))
    category_items["category"] = (category_items["product_category_name_english"]
                                  .fillna(category_items["product_category_name"]).fillna("unknown"))
    category = (category_items.groupby("category", as_index=False)
                .agg(item_count=("order_item_id", "size"), distinct_orders=("order_id", "nunique"),
                     product_value_brl=("price", "sum"))
                .sort_values("product_value_brl", ascending=False).reset_index(drop=True))
    category["value_share_pct"] = category["product_value_brl"] / category["product_value_brl"].sum() * 100

    state = (sales.groupby("customer_state", dropna=False, as_index=False)
             .agg(delivered_orders=("order_id", "nunique"), product_value_brl=("product_value", "sum"),
                  unique_customers=("customer_unique_id", "nunique"))
             .sort_values("product_value_brl", ascending=False).reset_index(drop=True))
    state["average_order_value_brl"] = state["product_value_brl"] / state["delivered_orders"]
    state["value_share_pct"] = state["product_value_brl"] / state["product_value_brl"].sum() * 100

    customer_base = sales.dropna(subset=["customer_unique_id"]).copy()
    customer_summary = (customer_base.groupby("customer_unique_id", as_index=False)
                        .agg(delivered_orders=("order_id", "nunique"), product_value_brl=("product_value", "sum")))
    frequency = (customer_summary["delivered_orders"].value_counts().sort_index()
                 .rename_axis("orders_per_customer").reset_index(name="customers"))
    repeat_customers = int((customer_summary["delivered_orders"] >= 2).sum())
    repeat_pct = 100 * repeat_customers / len(customer_summary) if len(customer_summary) else 0.0

    delivery = sales[["order_id", "customer_state", "order_purchase_timestamp",
                      "order_delivered_customer_date", "order_estimated_delivery_date"]].copy()
    for column in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        delivery[column] = pd.to_datetime(delivery[column], errors="coerce")
    delivery_valid = delivery.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date",
                                             "order_estimated_delivery_date"]).copy()
    delivery_valid["delivery_days"] = ((delivery_valid["order_delivered_customer_date"] -
                                        delivery_valid["order_purchase_timestamp"]).dt.total_seconds() / 86400)
    invalid_duration = int((delivery_valid["delivery_days"] < 0).sum())
    delivery_valid = delivery_valid.loc[delivery_valid["delivery_days"] >= 0].copy()
    delivery_valid["is_late"] = (delivery_valid["order_delivered_customer_date"].dt.normalize() >
                                 delivery_valid["order_estimated_delivery_date"].dt.normalize())
    state_delivery = (delivery_valid.groupby("customer_state", dropna=False, as_index=False)
                      .agg(orders=("order_id", "nunique"), late_orders=("is_late", "sum"),
                           median_delivery_days=("delivery_days", "median")))
    state_delivery["late_pct"] = 100 * state_delivery["late_orders"] / state_delivery["orders"]
    state_delivery = state_delivery.sort_values("customer_state").reset_index(drop=True)
    late_count = int(delivery_valid["is_late"].sum())
    valid_count = int(len(delivery_valid))
    late_pct = 100 * late_count / valid_count if valid_count else None

    monthly_base = sales.copy()
    monthly_base["purchase_date"] = pd.to_datetime(monthly_base["order_purchase_timestamp"], errors="coerce")
    missing_purchase_dates = int(monthly_base["purchase_date"].isna().sum())
    monthly_base = monthly_base.dropna(subset=["purchase_date", "product_value"]).copy()
    monthly_base["month"] = monthly_base["purchase_date"].dt.to_period("M").astype(str)
    monthly = (monthly_base.groupby("month", as_index=False)
               .agg(value_brl=("product_value", "sum"), orders=("order_id", "nunique"))
               .sort_values("month").reset_index(drop=True))

    # Reconciliation checks prevent double-counted orders after joining tables.
    assert not category_items.duplicated(subset=["order_id", "order_item_id"]).any()
    assert abs(category["product_value_brl"].sum() - expected) < 0.01
    assert abs(state["product_value_brl"].sum() - expected) < 0.01
    assert int(state["delivered_orders"].sum()) == len(sales)
    assert int(customer_summary["delivered_orders"].sum()) == len(customer_base)
    assert delivery_valid["order_id"].is_unique
    assert int(state_delivery["orders"].sum()) == valid_count
    assert int(state_delivery["late_orders"].sum()) == late_count
    if missing_purchase_dates:
        raise ValueError(f"{missing_purchase_dates} delivered orders lack a valid purchase date. "
                         "The month-level totals would not match the overall totals; inspect the raw data.")
    assert abs(monthly["value_brl"].sum() - expected) < 0.01
    assert int(monthly["orders"].sum()) == len(sales)

    metrics = {
        "source": "Olist Brazilian E-Commerce Public Dataset",
        "scope": "delivered orders with item records",
        "currency": "BRL",
        "total_orders_all_statuses": int(len(order_level)),
        "delivered_orders_with_items": int(len(sales)),
        "delivered_item_value_brl": round(expected, 2),
        "average_delivered_order_item_value_brl": round(expected / len(sales), 2),
        "repeat_customers": repeat_customers,
        "repeat_customer_percentage": round(repeat_pct, 2),
        "valid_delivery_records": valid_count,
        "late_deliveries": late_count,
        "late_delivery_percentage": round(late_pct, 2) if valid_count else None,
        "excluded_negative_delivery_durations": invalid_duration,
        "note": "Item value excludes freight; figures are historical, not realized revenue.",
    }
    reports = {
        "category_performance.csv": category,
        "state_performance.csv": state,
        "customer_order_frequency.csv": frequency,
        "delivery_by_state.csv": state_delivery,
        "monthly_performance.csv": monthly,
    }
    return reports, metrics


def save_reports(reports_dir: Path, reports: dict[str, pd.DataFrame], metrics: dict) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for filename, dataframe in reports.items():
        dataframe.to_csv(reports_dir / filename, index=False, encoding="utf-8-sig")
    (reports_dir / "metrics_snapshot.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(reports)+1} analytics reports to {reports_dir}")


# %% 04 — Load reports / repeatable consistency checks

def load_existing_reports(reports_dir: Path) -> dict:
    missing = [name for name in REQUIRED_REPORTS if not (reports_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing report(s) in {reports_dir}: {missing}")
    result = {
        "categories": pd.read_csv(reports_dir / "category_performance.csv"),
        "states": pd.read_csv(reports_dir / "state_performance.csv"),
        "frequency": pd.read_csv(reports_dir / "customer_order_frequency.csv"),
        "delivery": pd.read_csv(reports_dir / "delivery_by_state.csv"),
        "monthly": pd.read_csv(reports_dir / "monthly_performance.csv"),
        "metrics": json.loads((reports_dir / "metrics_snapshot.json").read_text(encoding="utf-8-sig")),
    }
    check_report_consistency(result)
    return result


def check_report_consistency(r: dict) -> None:
    m = r["metrics"]
    total = float(m["delivered_item_value_brl"])
    for name in ("categories", "states"):
        observed = float(r[name]["product_value_brl"].sum())
        if abs(observed - total) >= 0.01:
            raise ValueError(f"Total mismatch in {name}: {observed} vs. {total}")
    if abs(float(r["monthly"]["value_brl"].sum()) - total) >= 0.01:
        raise ValueError("Monthly item value does not reconcile to the total.")
    if int(r["states"]["delivered_orders"].sum()) != int(m["delivered_orders_with_items"]):
        raise ValueError("State-level order counts do not reconcile.")
    if int((r["frequency"]["orders_per_customer"] * r["frequency"]["customers"]).sum()) != int(m["delivered_orders_with_items"]):
        raise ValueError("Customer order-frequency counts do not reconcile; check missing unique customer IDs.")
    if int(r["delivery"]["orders"].sum()) != int(m["valid_delivery_records"]):
        raise ValueError("Delivery counts do not reconcile.")
    if int(r["delivery"]["late_orders"].sum()) != int(m["late_deliveries"]):
        raise ValueError("Late-delivery counts do not reconcile.")
    if int(r["monthly"]["orders"].sum()) != int(m["delivered_orders_with_items"]):
        raise ValueError("Monthly orders do not reconcile.")
    print("All report consistency checks passed.")


# %% 05 — Render the fully offline interactive HTML dashboard
# This is the full HTML/CSS/JavaScript template, adapted from the validated
# report in this project. Data values are inserted from CSV/JSON outputs.
DASHBOARD_TEMPLATE = r'''<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Interactive portfolio analysis of the historical Olist Brazilian E-Commerce dataset, with independently validated category, geographic, customer and delivery metrics.">
<title>OLIST | داشبورد جامع تحلیل تجارت الکترونیک</title>
<style>
:root{--bg:#f4f7fc;--card:#fff;--ink:#203451;--navy:#172a46;--line:#e4ebf5;--muted:#7889a0;--teal:#0f9b8f;--indigo:#5868d6;--violet:#8669bd;--orange:#ecab5b;--red:#d96076;--shadow:0 12px 35px rgba(27,53,85,.055)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:Tahoma,"Segoe UI",Arial,sans-serif;font-size:13px}button,select{font:inherit}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid rgba(88,104,214,.35);outline-offset:3px}a{text-decoration:none;color:inherit}
.shell{width:min(1540px,100%);margin:auto;padding:30px 30px 48px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap}.brand{font:800 11px Arial;letter-spacing:2px;color:#657ca1;direction:ltr;text-align:right}.brand strong{background:var(--navy);color:#fff;display:inline-grid;place-items:center;width:38px;height:38px;border-radius:11px;margin-left:8px;font:800 22px Arial;letter-spacing:-1px}h1{font-size:28px;margin:15px 0 8px;line-height:1.65}h2{font-size:16px;margin:0}p{line-height:2}.lede{color:var(--muted);max-width:850px;margin:0}.pills{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-start}.pill{border:1px solid #dae4f1;border-radius:99px;background:#fff;color:#607693;padding:9px 12px;font-size:11px}.pill span{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--orange);margin-left:5px}
.nav{display:flex;gap:8px;flex-wrap:wrap;margin:25px 0 18px}.nav a{border:1px solid var(--line);padding:9px 14px;border-radius:10px;background:white;font-weight:700;font-size:11px;color:#526986}.nav a:hover{background:#eaf0fc;color:#354b82}
.kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:23px}.kpi{border-radius:17px;background:white;border:1px solid var(--line);box-shadow:var(--shadow);padding:19px 21px;min-width:0;position:relative;overflow:hidden}.kpi.feature{background:linear-gradient(120deg,#1f3b60,#172a46);color:white;border-color:#172a46}.kpi:not(.feature):before{content:"";position:absolute;right:0;top:0;height:100%;width:4px;background:var(--teal)}.kpi:nth-child(3):before{background:var(--indigo)}.kpi:nth-child(4):before{background:var(--violet)}.kpi:nth-child(5):before{background:var(--orange)}.kpi:nth-child(6):before{background:var(--red)}.kpi .label{font-weight:700;color:#6d809a;font-size:12px;line-height:1.7}.kpi.feature .label{color:#cad8e9}.kpi .n{direction:ltr;text-align:right;font:bold clamp(22px,2.3vw,30px) Arial;margin:16px 0 9px;font-variant-numeric:tabular-nums}.kpi .detail{color:#8b9bb0;font-size:10.5px;line-height:1.7}.kpi.feature .detail{color:#c7d5e7}
.section{margin-top:26px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:0 2px 14px}.eyebrow{font:700 10px Arial;letter-spacing:1.4px;color:#8c9cb5;direction:ltr;text-align:right;margin-bottom:7px}.section-head p{margin:3px 0 0;color:#8595ab;font-size:11px}.context{border:1px solid #e2eaf3;color:#6f839f;background:#fff;border-radius:99px;padding:7px 12px;font-size:10px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.card{background:white;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden;min-width:0}.card-inner{padding:20px 20px 8px}.card h3{font-size:13px;margin:0 0 5px;color:#2e4162}.card .sub{font-size:10.5px;color:#94a1b3;line-height:1.8}.chart{height:340px;direction:ltr}.chart.small{height:300px}.chart.tall{height:390px}.chart-note{border-top:1px solid #eff2f6;padding:13px 20px 15px;font-size:10.5px;color:#8496aa;line-height:1.85}.card.wide{grid-column:1 / -1}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:15px}.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.select,.btn{border:1px solid #dbe4f0;border-radius:10px;background:#fff;padding:10px 12px;font-size:11px;color:#345076;cursor:pointer}.btn{background:#172a46;color:#fff;border-color:#172a46;font-weight:700}.help{color:#8a9bb0;font-size:11px}
.insights{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.insight{padding:20px 21px;background:white;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.insight .marker{display:inline-block;font:800 10px Arial;letter-spacing:1px;color:var(--indigo);margin-bottom:10px}.insight b{display:block;font-size:15px;line-height:1.7;margin-bottom:7px}.insight p{margin:0;color:#657a96;font-size:11.5px}.footer-note{border:1px solid #dfe7f2;border-radius:14px;background:#fdfefe;padding:16px 20px;margin-top:23px;color:#71849e;font-size:11px;line-height:2}.footer-note b{color:#3b5170}.footer{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:17px;color:#9aacc0;font:10px Arial,Tahoma}
@media(max-width:970px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.cards,.insights{grid-template-columns:1fr}.card.wide{grid-column:auto}.shell{padding:22px 14px 35px}}@media(max-width:560px){h1{font-size:23px}.kpi{padding:14px}.kpi .n{font-size:21px}.card-inner{padding:15px 9px 6px}.chart,.chart.tall{height:340px}.chart.small{height:275px}.nav a{padding:8px 10px}.insight{padding:16px}.pills{justify-content:flex-start}}
</style></head><body>
<div class="shell"><header class="top"><div><div class="brand">OLIST · BUSINESS INTELLIGENCE <strong>O</strong></div><h1>داشبورد جامع تحلیل تجارت الکترونیک</h1><p class="lede">تحلیل داده‌های تاریخی سفارش‌های تحویل‌شده Olist؛ بررسی ارزش کالا، عملکرد محصولات، تمرکز جغرافیایی، الگوی سفارش مشتریان و کیفیت تحویل.</p></div><div class="pills"><span class="pill"><span></span> داده تاریخی، نه زنده</span><span class="pill">BRL · بدون هزینه ارسال</span><span class="pill">__RANGE__</span></div></header>
<nav class="nav"><a href="#overview">شاخص‌های اصلی</a><a href="#sales">روند ماهانه</a><a href="#product">محصولات</a><a href="#geography">جغرافیا</a><a href="#customers">مشتریان</a><a href="#delivery">تحویل</a><a href="#insights">نتایج مدیریتی</a></nav>
<section class="kpis" id="overview"><div class="kpi feature"><div class="label">ارزش کالاهای تحویل‌شده</div><div class="n" id="total-value"></div><div class="detail">BRL | مجموع قیمت اقلام، نه سود یا درآمد وصول‌شده</div></div><div class="kpi"><div class="label">سفارش‌های تحویل‌شده</div><div class="n" id="total-orders"></div><div class="detail">سفارش یکتا دارای اقلام ثبت‌شده</div></div><div class="kpi"><div class="label">میانگین ارزش کالا در سفارش</div><div class="n" id="avg-value"></div><div class="detail">BRL | ارزش کالا ÷ تعداد سفارش</div></div><div class="kpi"><div class="label">مشتریان یکتای این تحلیل</div><div class="n" id="customer-count"></div><div class="detail">مشتریانی با سفارش تحویل‌شده دارای اقلام</div></div><div class="kpi"><div class="label">مشتریان دارای سفارش تکراری</div><div class="n" id="repeat-rate"></div><div class="detail">درصد مشتریان با بیش از یک سفارش</div></div><div class="kpi"><div class="label">نرخ تحویل دیرتر از برآورد</div><div class="n" id="late-rate"></div><div class="detail">بر مبنای سفارش‌های دارای اطلاعات تحویل معتبر</div></div></section>
<section class="section" id="sales"><div class="section-head"><div><div class="eyebrow">01 / SALES OVERVIEW</div><h2>روند تاریخی ارزش کالا و سفارش‌ها</h2><p>فیلتر این بخش تنها روی دو نمودار ماهانه اثر دارد؛ سایر شاخص‌ها همیشه مربوط به کل دوره‌اند.</p></div><div class="controls"><label for="period" class="help">بازه نمودار</label><select id="period" class="select"><option value="all">کل دوره</option><option value="2017">سال ۲۰۱۷</option><option value="2018">سال ۲۰۱۸ (تا اوت)</option><option value="latest12">۱۲ ماه آخر</option></select><button type="button" id="csv" class="btn">دریافت داده ماهانه CSV</button></div></div><div class="cards"><article class="card"><div class="card-inner"><h3>ارزش ماهانه کالاهای تحویل‌شده</h3><div class="sub">BRL · ماه ثبت سفارش · دوره انتخاب‌شده</div><div class="chart" id="ch-monthly-value"></div></div><div class="chart-note" id="monthly-note">ماه‌های فاقد رکورد خالی نمایش داده می‌شوند؛ ارزش کالا هزینه ارسال را شامل نمی‌شود.</div></article><article class="card"><div class="card-inner"><h3>تعداد سفارش‌های ماهانه</h3><div class="sub">تعداد سفارش یکتا · دوره انتخاب‌شده</div><div class="chart" id="ch-monthly-orders"></div></div><div class="chart-note">ماه‌های کم‌حجم تاریخی در این نمودار حذف نشده‌اند.</div></article></div></section>
<section class="section" id="product"><div class="section-head"><div><div class="eyebrow">02 / PRODUCT MIX</div><h2>کدام گروه‌های محصول سهم بیشتری دارند؟</h2><p>۱۰ دسته برتر براساس ارزش کالا؛ اسامی همان کدهای دسته‌بندی در داده اصلی هستند.</p></div><span class="context">کل دوره · ۷۴ دسته محصول</span></div><div class="cards"><article class="card wide"><div class="card-inner"><h3>۱۰ دسته محصول برتر</h3><div class="sub">ارزش کالا برحسب BRL · درصد سهم از کل در جزئیات نمودار</div><div class="chart tall" id="ch-categories"></div></div><div class="chart-note">شمارش آیتم‌ها و سفارش‌های متمایز نیز در جزئیات هر ستون ارائه شده است.</div></article></div></section>
<section class="section" id="geography"><div class="section-head"><div><div class="eyebrow">03 / GEOGRAPHIC PERFORMANCE</div><h2>تمرکز جغرافیایی ارزش کالا</h2><p>تفکیک براساس ایالت مشتری؛ نه ایالت فروشنده یا مکان انبار.</p></div><span class="context">کل دوره · ۲۷ ایالت</span></div><div class="cards"><article class="card wide"><div class="card-inner"><h3>۱۰ ایالت برتر از نظر ارزش کالا</h3><div class="sub">کد ایالت‌های برزیل · BRL · تعداد سفارش و سهم در جزئیات نمودار</div><div class="chart tall" id="ch-states"></div></div><div class="chart-note">تفاوت بین اندازه بازار و عملکرد نسبی باید با توجه به تعداد سفارش‌ها تفسیر شود.</div></article></div></section>
<section class="section" id="customers"><div class="section-head"><div><div class="eyebrow">04 / CUSTOMER BEHAVIOR</div><h2>مشتریان چند بار سفارش داده‌اند؟</h2><p>فراوانی تعداد سفارش‌های تحویل‌شده به ازای شناسه یکتای مشتری.</p></div><span class="context">کل دوره · ۹۳٬۳۵۸ مشتری</span></div><div class="cards"><article class="card"><div class="card-inner"><h3>مشتریان تک‌سفارشی در برابر تکراری</h3><div class="sub">سهم مشتریان در بازه داده تاریخی</div><div class="chart small" id="ch-repeat"></div></div><div class="chart-note">این معیار با «نرخ ماندگاری مشتری در یک دوره زمانی مشخص» متفاوت است.</div></article><article class="card"><div class="card-inner"><h3>توزیع تعداد سفارش برای هر مشتری</h3><div class="sub">تعداد مشتری به تفکیک دفعات سفارش</div><div class="chart small" id="ch-frequency"></div></div><div class="chart-note">برای خوانایی محور، مقیاس تعداد مشتریان لگاریتمی است.</div></article></div></section>
<section class="section" id="delivery"><div class="section-head"><div><div class="eyebrow">05 / DELIVERY QUALITY</div><h2>بررسی تحقق زمان تحویل</h2><p>مقایسه تاریخ واقعی تحویل با تاریخ تحویل تخمینی در سفارش‌های قابل ارزیابی.</p></div><span class="context">کل دوره · ۹۶٬۴۷۰ سفارش معتبر</span></div><div class="cards"><article class="card"><div class="card-inner"><h3>سفارش‌های به‌موقع و دیرهنگام</h3><div class="sub">مجموع سفارش‌های دارای اطلاعات تحویل معتبر</div><div class="chart small" id="ch-delivery"></div></div><div class="chart-note">۸ سفارش تحویل‌شده، اطلاعات لازم برای این محاسبه را نداشته‌اند.</div></article><article class="card"><div class="card-inner"><h3>ایالت‌های دارای نرخ تأخیر بالاتر</h3><div class="sub">فقط ایالت‌های دارای حداقل ۳۰۰ سفارش معتبر · درصد</div><div class="chart small" id="ch-late-states"></div></div><div class="chart-note">آستانه ۳۰۰ سفارش برای جلوگیری از بزرگ‌نمایی نرخ‌های نمونه‌های بسیار کوچک انتخاب شده است.</div></article></div></section>
<section class="section" id="insights"><div class="section-head"><div><div class="eyebrow">06 / EXECUTIVE TAKEAWAYS</div><h2>یافته‌های مدیریتی و اقدام‌های پیشنهادی</h2><p>این پیشنهادها فرضیه‌های قابل بررسی هستند، نه نتیجه‌گیری علّی.</p></div></div><div class="insights" id="insight-cards"></div></section>
<div class="footer-note"><b>تعریف شاخص‌ها و محدودیت‌ها:</b> دامنه تحلیل سفارش‌های تحویل‌شده دارای رکورد کالا است. «ارزش کالا» مجموع price اقلام و بدون freight است؛ نه سود، نه درآمد وصول‌شده. نرخ تکرار خرید فقط در پنجره تاریخی موجود محاسبه شده، نه به‌صورت cohort retention. نرخ تأخیر تنها بر روی سفارش‌های دارای زمان واقعی و برآورد تحویل معتبر محاسبه می‌شود. نمودارهای دسته‌بندی، جغرافیا، مشتری و تحویل همیشه کل دوره را نمایش می‌دهند. منبع: Olist Brazilian E-Commerce Public Dataset. خروجی از جداول تجمیعی نوت‌بوک تهیه شده است.</div><footer class="footer"><span>OLIST · DATA ANALYTICS PORTFOLIO</span><span>Self-contained interactive report · validated aggregates</span></footer></div>
<script id="data" type="application/json">__DATA__</script>
<script>__PLOTLY__</script>
<script>
(()=>{'use strict';
const D=JSON.parse(document.getElementById('data').textContent),M=D.metrics;
const n=x=>Number(x),sum=(a,f)=>a.reduce((v,x)=>v+f(x),0),f0=new Intl.NumberFormat('en-US',{maximumFractionDigits:0}),f1=new Intl.NumberFormat('en-US',{maximumFractionDigits:1}),f2=new Intl.NumberFormat('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const el=id=>document.getElementById(id);const num=v=>f0.format(v);const cur=v=>f2.format(v);const pct=v=>f2.format(v)+'%';
const totalCustomers=sum(D.frequency,r=>n(r.customers));
const font={family:'Tahoma, Arial',size:11,color:'#62758f'},grid='#edf1f7',axis={showgrid:true,gridcolor:grid,zeroline:false,tickfont:{size:10,color:'#71819a'},showline:false};
const cfg={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d'],toImageButtonOptions:{format:'png',scale:2}};
const base=(left=65,bottom=50)=>({paper_bgcolor:'#fff',plot_bgcolor:'#fff',font,margin:{l:left,r:24,t:18,b:bottom},showlegend:false,hoverlabel:{bgcolor:'#243854',font:{family:'Tahoma, Arial',size:11,color:'#fff'}},xaxis:{...axis},yaxis:{...axis}});
const chart=(id,traces,layout)=>Plotly.newPlot(id,traces,layout,cfg);
el('total-value').textContent=cur(M.delivered_item_value_brl);
el('total-orders').textContent=num(M.delivered_orders_with_items);
el('avg-value').textContent=cur(M.average_delivered_order_item_value_brl);
el('customer-count').textContent=num(totalCustomers);
el('repeat-rate').textContent=pct(M.repeat_customer_percentage);
el('late-rate').textContent=pct(M.late_delivery_percentage);
let monthlyRows=[];
function renderMonthly(){
 const period=el('period').value;
 const latestMonth=D.monthly[D.monthly.length-1].month;
 const [latestYear,latestMonthNumber]=latestMonth.split('-').map(Number);
 const last12Start=new Date(Date.UTC(latestYear,latestMonthNumber-12,1)).toISOString().slice(0,7);
 const rows=D.monthly.filter(r=>period==='all'||(period==='latest12'?r.month>=last12Start:r.month.startsWith(period)));
 monthlyRows=rows;
 const months=[];
 if(rows.length){let [year,month]=rows[0].month.split('-').map(Number);const end=rows[rows.length-1].month;
  for(let guard=0;guard<240;guard++){const key=String(year)+'-'+String(month).padStart(2,'0');months.push(key);if(key===end)break;month++;if(month>12){month=1;year++;}}}
 const lookup=new Map(rows.map(r=>[r.month,r]));
 const values=months.map(k=>lookup.has(k)?n(lookup.get(k).value_brl):null),orders=months.map(k=>lookup.has(k)?n(lookup.get(k).orders):null);
 const l=base(76,72);l.xaxis={...axis,tickangle:-45,type:'category'};l.yaxis={...axis,tickformat:',.0f'};
 chart('ch-monthly-value',[{type:'scatter',mode:'lines+markers',x:months,y:values,connectgaps:false,line:{color:'#5868d6',width:3},marker:{size:7,color:'#5868d6'},fill:'tozeroy',fillcolor:'rgba(88,104,214,.09)',hovertemplate:'%{x}<br>ارزش کالا: %{y:,.2f} BRL<extra></extra>'}],l);
 const l2=base(70,72);l2.xaxis={...axis,tickangle:-45,type:'category'};
 chart('ch-monthly-orders',[{type:'bar',x:months,y:orders,marker:{color:'#16a297',opacity:.95},hovertemplate:'%{x}<br>سفارش‌ها: %{y:,}<extra></extra>'}],l2);
 const highest=rows.reduce((a,r)=>!a||n(r.value_brl)>n(a.value_brl)?r:a,null);
 el('monthly-note').textContent=`${rows.length} ماه دارای رکورد | ارزش کالا: ${cur(sum(rows,r=>n(r.value_brl)))} BRL | بیشترین ماه: ${highest.month} با ${cur(n(highest.value_brl))} BRL. بازه فقط این دو نمودار را تغییر می‌دهد.`;
}
renderMonthly();el('period').addEventListener('change',renderMonthly);
function download(rows){const csv='\ufeff'+[['month','item_value_brl','delivered_orders'],...rows.map(r=>[r.month,n(r.value_brl).toFixed(2),r.orders])].map(r=>r.join(',')).join('\r\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='olist_monthly_'+el('period').value+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1500)}
el('csv').addEventListener('click',()=>download(monthlyRows));
const cats=D.categories.slice(0,10).reverse();
const barCats=base(175,50);barCats.xaxis={...axis,tickformat:',.0f'};barCats.yaxis={showgrid:false,zeroline:false,tickfont:{size:10,color:'#48607c'},automargin:true};
chart('ch-categories',[{type:'bar',orientation:'h',y:cats.map(r=>r.category.replaceAll('_',' ')),x:cats.map(r=>n(r.product_value_brl)),marker:{color:cats.map((r,i)=>i===9?'#5868d6':'#16a297')},customdata:cats.map(r=>[n(r.value_share_pct),n(r.distinct_orders),n(r.item_count)]),hovertemplate:'دسته: %{y}<br>ارزش: %{x:,.2f} BRL<br>سهم: %{customdata[0]:.2f}%<br>سفارش یکتا: %{customdata[1]:,}<br>تعداد اقلام: %{customdata[2]:,}<extra></extra>'}],barCats);
const st=D.states.slice(0,10).reverse();const barStates=base(62,50);barStates.xaxis={...axis,tickformat:',.0f'};barStates.yaxis={showgrid:false,zeroline:false,tickfont:{size:11,color:'#48607c'}};
chart('ch-states',[{type:'bar',orientation:'h',y:st.map(r=>r.customer_state),x:st.map(r=>n(r.product_value_brl)),marker:{color:st.map((r,i)=>i===9?'#5868d6':'#20a79d')},customdata:st.map(r=>[n(r.value_share_pct),n(r.delivered_orders),n(r.unique_customers)]),hovertemplate:'ایالت: %{y}<br>ارزش کالا: %{x:,.2f} BRL<br>سهم: %{customdata[0]:.2f}%<br>سفارش‌ها: %{customdata[1]:,}<br>مشتریان یکتا: %{customdata[2]:,}<extra></extra>'}],barStates);
function donut(id,labels,vals,colors){const lo=base(15,15);lo.margin={l:18,r:18,t:10,b:10};lo.showlegend=true;lo.legend={orientation:'h',x:.5,xanchor:'center',y:-.05,font:{size:11}};chart(id,[{type:'pie',hole:.69,labels,values:vals,sort:false,marker:{colors,line:{color:'#fff',width:3}},textinfo:'percent',textposition:'outside',hovertemplate:'%{label}<br>تعداد: %{value:,}<br>سهم: %{percent}<extra></extra>'}],lo)}
const singles=n(D.frequency.find(r=>n(r.orders_per_customer)===1).customers);
donut('ch-repeat',['یک سفارش','بیش از یک سفارش'],[singles,totalCustomers-singles],['#dfe7f2','#5868d6']);
const fq=D.frequency;const fl=base(60,52);fl.xaxis={...axis,title:{text:'دفعات سفارش',font:{size:10}},type:'category'};fl.yaxis={...axis,type:'log',title:{text:'تعداد مشتری (مقیاس لگاریتمی)',font:{size:10}}};
chart('ch-frequency',[{type:'bar',x:fq.map(r=>r.orders_per_customer),y:fq.map(r=>n(r.customers)),marker:{color:'#8669bd'},hovertemplate:'دفعات سفارش: %{x}<br>مشتریان: %{y:,}<extra></extra>'}],fl);
donut('ch-delivery',['به‌موقع یا زودتر','پس از موعد تخمینی'],[M.valid_delivery_records-M.late_deliveries,M.late_deliveries],['#10a395','#db6277']);
const eligibleLate=D.delivery.filter(r=>n(r.orders)>=300);
const late=(eligibleLate.length?eligibleLate:D.delivery.filter(r=>n(r.orders)>0)).sort((a,b)=>n(b.late_pct)-n(a.late_pct)).slice(0,8).reverse();
const ll=base(58,50);ll.xaxis={...axis,ticksuffix:'%',range:[0,Math.max(24,Math.ceil(n(late[late.length-1]?.late_pct||0)/5)*5)]};ll.yaxis={showgrid:false,zeroline:false,tickfont:{size:11,color:'#48607c'}};
chart('ch-late-states',[{type:'bar',orientation:'h',x:late.map(r=>n(r.late_pct)),y:late.map(r=>r.customer_state),marker:{color:late.map((r,i)=>i===7?'#d96076':'#ecab5b')},customdata:late.map(r=>[n(r.orders),n(r.late_orders),n(r.median_delivery_days)]),hovertemplate:'ایالت: %{y}<br>تأخیر: %{x:.2f}%<br>سفارش معتبر: %{customdata[0]:,}<br>دیرهنگام: %{customdata[1]:,}<br>میانه روزهای تحویل: %{customdata[2]:.1f}<extra></extra>'}],ll);
const topCat=D.categories[0],topSt=D.states[0],top3share=sum(D.states.slice(0,3),r=>n(r.value_share_pct));
const topLate=(eligibleLate.length?eligibleLate:D.delivery.filter(r=>n(r.orders)>0)).sort((a,b)=>n(b.late_pct)-n(a.late_pct))[0];
const insights=[
['PRODUCT PORTFOLIO',`دسته ${topCat.category.replaceAll('_',' ')} در رتبه اول`,`سهم این دسته از ارزش کالا ${pct(n(topCat.value_share_pct))} است. می‌توان ترکیب دسته‌های پرفروش را دقیق‌تر از نظر رشد، تعداد سفارش و میانگین ارزش هر سفارش بررسی کرد.`],
['GEOGRAPHIC CONCENTRATION',`${pct(top3share)} ارزش کالا در سه ایالت نخست`,`ایالت ${topSt.customer_state} به‌تنهایی ${pct(n(topSt.value_share_pct))} سهم دارد. برای برنامه‌ریزی بازاریابی، اندازه بازارهای بزرگ و امکان توسعه در ایالت‌های کم‌سهم باید جداگانه تحلیل شود.`],
['CUSTOMER PATTERN',`${pct(M.repeat_customer_percentage)} از مشتریان چند سفارش داشته‌اند`,`تنها ${num(M.repeat_customers)} مشتری بیش از یک سفارش تحویل‌شده در این بازه دارند. برای سنجش وفاداری واقعی، تحلیل کوهورت با کنترل زمان اولین خرید لازم است.`],
['DELIVERY PRIORITY',`${pct(n(topLate.late_pct))} تأخیر در ایالت ${topLate.customer_state}`,`این نرخ از ${num(n(topLate.orders))} سفارش معتبر به دست آمده است. بررسی هم‌زمان نرخ و تعداد دیرکردها و سپس مقایسه مسیرها و فروشندگان، یک گام تحلیلی بعدی است.`]
];
el('insight-cards').innerHTML=insights.map((r,i)=>`<article class="insight"><span class="marker">${String(i+1).padStart(2,'0')} / ${r[0]}</span><b>${r[1]}</b><p>${r[2]}</p></article>`).join('');
})();
</script></body></html>
'''


def write_dashboard(reports_dir: Path, output_file: Path) -> Path:
    results = load_existing_reports(reports_dir)
    # Keep numerical and string data types read from the user's reports.
    data = {
        "metrics": results["metrics"],
        "categories": results["categories"].to_dict(orient="records"),
        "states": results["states"].to_dict(orient="records"),
        "frequency": results["frequency"].to_dict(orient="records"),
        "delivery": results["delivery"].to_dict(orient="records"),
        "monthly": results["monthly"].to_dict(orient="records"),
    }
    # Protect embedded JSON from closing its script tag.
    json_blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    json_blob = json_blob.replace("<", "\\u003c")
    js_blob = get_plotlyjs()  # Plotly ships the JS library; dashboard stays offline.
    first_month = str(results["monthly"]["month"].iloc[0]).replace("-", "/")
    last_month = str(results["monthly"]["month"].iloc[-1]).replace("-", "/")
    html = (DASHBOARD_TEMPLATE.replace("__DATA__", json_blob)
            .replace("__PLOTLY__", js_blob)
            .replace("__RANGE__", f"{first_month} تا {last_month}"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    print(f"Interactive dashboard saved: {output_file} ({output_file.stat().st_size/1024/1024:.1f} MB)")
    return output_file


# %% 06 — Run end-to-end and create a ZIP for convenient download

def run_pipeline(
    data_dir: str | Path | None = None,
    output_dir: str | Path = "olist_output",
    reports_dir: str | Path | None = None,
) -> Path:
    """Execute either from Kaggle/local raw CSVs or from an existing reports directory."""
    root = Path(output_dir).expanduser().resolve()
    output_reports = root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    if reports_dir is not None:
        source_reports = Path(reports_dir).expanduser().resolve()
        if source_reports != output_reports.resolve():
            output_reports.mkdir(parents=True, exist_ok=True)
            for filename in REQUIRED_REPORTS:
                shutil.copy2(source_reports / filename, output_reports / filename)
        else:
            output_reports = source_reports
        print("Using precomputed reports:", output_reports)
    else:
        source = get_dataset_directory(data_dir)
        tables = read_raw_tables(source)
        primary, relations = validate_tables(tables)
        order_level = make_order_level(tables)
        reports, metrics = analyze_olist(tables, order_level)
        save_reports(output_reports, reports, metrics)
        primary.to_csv(output_reports / "key_validation.csv", index=False, encoding="utf-8-sig")
        relations.to_csv(output_reports / "relation_validation.csv", index=False, encoding="utf-8-sig")
        make_status_report(order_level).to_csv(output_reports / "order_status_summary.csv", index=False, encoding="utf-8-sig")
        print("Data quality and order-status audit reports saved.")

    html_path = write_dashboard(output_reports, root / "Olist_Portfolio_Executive_Dashboard.html")
    zip_file = shutil.make_archive(str(root.parent / (root.name + "_files")), "zip", root)
    print("ZIP archive saved:", zip_file)
    return html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Olist analytics and dashboard pipeline.")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing raw Olist CSVs; omitted = download via Kagglehub")
    parser.add_argument("--reports-dir", type=str, default=None, help="Use existing aggregate CSV/JSON files instead of raw data")
    parser.add_argument("--output-dir", type=str, default="olist_output", help="Where to write HTML and CSV/JSON outputs")
    args = parser.parse_args()
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, reports_dir=args.reports_dir)
