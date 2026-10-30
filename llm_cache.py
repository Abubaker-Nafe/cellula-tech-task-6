from langchain_redis import RedisCache

from cache_monitor import record_cache_lookup

class LoggedRedisCache(RedisCache):
    def __init__(self, *args, validate_output=None, validate_generation=None, **kwargs):
        self.validate_output = validate_output
        self.validate_generation = validate_generation
        super().__init__(*args, **kwargs)

    def _is_valid(self, generations):
        if not generations:
            return False
        try:
            for generation in generations:
                if self.validate_generation is not None:
                    self.validate_generation(generation)
                if self.validate_output is not None:
                    self.validate_output(str(generation.text))
        except (ValueError, TypeError, AttributeError):
            return False
        return True

    def lookup(self, prompt, llm_string):
        result = super().lookup(prompt, llm_string)
        if result is not None and not self._is_valid(result):
            print("Cache: MISS (invalid cached output ignored)")
            record_cache_lookup("MISS")
            return None
        status = "HIT" if result is not None else "MISS"
        print("Cache:", status)
        record_cache_lookup(status)
        return result

    def update(self, prompt, llm_string, return_val):
        if not self._is_valid(return_val):
            print("Cache: SKIP (invalid output)")
            return
        super().update(prompt, llm_string, return_val)
