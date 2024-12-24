import os

class BBGFields:
    def __init__(self):
        self._FILING_STATUS_PRIORITY = {
            'MR': 1,
            'OR': 2,
            'PR': 3,
            'RS': 4,
            'ER': 5
        }
        self._ACCOUNTING_STANDARD_PRIORITY = {
            'IAS/IFRS': 1,
            'US GAAP': 2
        }

    @property
    def FILING_STATUS_PRIORITY(self):
        return self._FILING_STATUS_PRIORITY

    @property
    def ACCOUNTING_STANDARD_PRIORITY(self):
        return self._ACCOUNTING_STANDARD_PRIORITY
