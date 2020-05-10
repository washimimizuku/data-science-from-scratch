from dataclasses import dataclass

import datetime

@dataclass
class StockPrice:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """It's a class, so we can add methods too"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()

# stock split
price.closing_price /= 2
assert price.closing_price == 53.015

# It's a regular class, so add new fields however you like!
price.cosing_price = 75 # oops
