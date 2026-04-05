from __future__ import annotations

import unittest

from hdb_resale_mlops.features import parse_remaining_lease_years, parse_storey_midpoint


class FeatureParsingTest(unittest.TestCase):
    def test_parse_remaining_lease_years_and_months(self) -> None:
        self.assertAlmostEqual(parse_remaining_lease_years("61 years 04 months"), 61.3333333333)

    def test_parse_remaining_lease_years_only(self) -> None:
        self.assertEqual(parse_remaining_lease_years("72 years"), 72.0)

    def test_parse_storey_midpoint(self) -> None:
        self.assertEqual(parse_storey_midpoint("04 TO 06"), 5.0)

