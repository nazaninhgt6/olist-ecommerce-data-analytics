"""Tests of the provided aggregate files, chart payload, and audit exports."""
import json
import re
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

class PublishedReportsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((REPORTS / 'metrics_snapshot.json').read_text(encoding='utf-8-sig'))
        cls.category = pd.read_csv(REPORTS / 'category_performance.csv')
        cls.state = pd.read_csv(REPORTS / 'state_performance.csv')
        cls.monthly = pd.read_csv(REPORTS / 'monthly_performance.csv')
        cls.frequency = pd.read_csv(REPORTS / 'customer_order_frequency.csv')
        cls.delivery = pd.read_csv(REPORTS / 'delivery_by_state.csv')

    def test_item_value_reconciles_across_aggregations(self):
        total = self.metrics['delivered_item_value_brl']
        for df, col in ((self.category, 'product_value_brl'), (self.state,'product_value_brl'), (self.monthly,'value_brl')):
            self.assertAlmostEqual(float(df[col].sum()), float(total), places=2)

    def test_order_counts_reconcile(self):
        target = self.metrics['delivered_orders_with_items']
        self.assertEqual(int(self.state['delivered_orders'].sum()), target)
        self.assertEqual(int(self.monthly['orders'].sum()), target)
        self.assertEqual(int((self.frequency.orders_per_customer * self.frequency.customers).sum()), target)

    def test_delivery_counts_reconcile(self):
        self.assertEqual(int(self.delivery.orders.sum()), int(self.metrics['valid_delivery_records']))
        self.assertEqual(int(self.delivery.late_orders.sum()), int(self.metrics['late_deliveries']))

    def test_audit_keys(self):
        key = pd.read_csv(REPORTS/'key_validation.csv')
        relations = pd.read_csv(REPORTS/'relation_validation.csv')
        self.assertTrue((key[['missing_keys','duplicate_keys']] == 0).all().all())
        self.assertTrue((relations[['missing_keys','unmatched_keys']] == 0).all().all())

    def test_status_total_and_missing_payment(self):
        status = pd.read_csv(REPORTS/'order_status_summary.csv')
        self.assertEqual(int(status.total_orders.sum()), int(self.metrics['total_orders_all_statuses']))
        delivered = status.loc[status.order_status=='delivered'].iloc[0]
        self.assertEqual(int(delivered.missing_payments), 1)

    def test_embedded_dashboard_data_match_reports(self):
        html = (ROOT/'docs/index.html').read_text(encoding='utf-8')
        match = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
        self.assertIsNotNone(match, 'Expected embedded self-contained report payload')
        payload = json.loads(match.group(1))
        self.assertEqual(payload['metrics']['delivered_orders_with_items'],self.metrics['delivered_orders_with_items'])
        self.assertAlmostEqual(payload['metrics']['delivered_item_value_brl'], self.metrics['delivered_item_value_brl'])
        self.assertEqual(sum(r['orders'] for r in payload['monthly']),self.metrics['delivered_orders_with_items'])
        self.assertEqual(sum(r['late_orders'] for r in payload['delivery']),self.metrics['late_deliveries'])
        self.assertEqual(len(payload['categories']),len(self.category))
        self.assertEqual(len(payload['states']),len(self.state))

if __name__ == '__main__':
    unittest.main()
