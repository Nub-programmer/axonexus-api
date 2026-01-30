import time
from typing import Dict, Tuple

class RateLimiter:
    def __init__(self):
        # {key: (timestamp, count)}
        self.requests: Dict[str, Tuple[float, int]] = {}
        # {key: (date, total_tokens)}
        self.usage: Dict[str, Tuple[str, int]] = {}
        
        # IP/Guest limits
        self.GUEST_RPM = 5
        self.GUEST_DAILY_TOKENS = 2000
        
        # Test key limits
        self.TEST_RPM = 10
        self.TEST_DAILY_TOKENS = 10000
        
        # Premium key limits
        self.PREMIUM_RPM = 30
        self.PREMIUM_DAILY_TOKENS = 50000

    def get_current_date(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def check_rate_limit(self, key: str, is_test: bool, is_guest: bool) -> bool:
        now = time.time()
        rpm_limit = self.PREMIUM_RPM
        if is_test: rpm_limit = self.TEST_RPM
        if is_guest: rpm_limit = self.GUEST_RPM
        
        if key not in self.requests:
            self.requests[key] = (now, 1)
            return True
            
        last_time, count = self.requests[key]
        if now - last_time > 60:
            self.requests[key] = (now, 1)
            return True
            
        if count < rpm_limit:
            self.requests[key] = (last_time, count + 1)
            return True
            
        return False

    def check_usage_limit(self, key: str, is_test: bool, is_guest: bool) -> bool:
        today = self.get_current_date()
        daily_limit = self.PREMIUM_DAILY_TOKENS
        if is_test: daily_limit = self.TEST_DAILY_TOKENS
        if is_guest: daily_limit = self.GUEST_DAILY_TOKENS
        
        if key not in self.usage:
            self.usage[key] = (today, 0)
            return True
            
        usage_date, tokens = self.usage[key]
        if usage_date != today:
            self.usage[key] = (today, 0)
            return True
            
        return tokens < daily_limit

    def update_usage(self, key: str, tokens: int):
        today = self.get_current_date()
        if key in self.usage:
            usage_date, current_tokens = self.usage[key]
            if usage_date == today:
                self.usage[key] = (today, current_tokens + tokens)
            else:
                self.usage[key] = (today, tokens)
        else:
            self.usage[key] = (today, tokens)

limiter = RateLimiter()

def get_limiter() -> RateLimiter:
    return limiter
