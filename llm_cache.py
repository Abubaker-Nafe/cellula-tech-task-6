import json

from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
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
        # JSONPath '$' returns a list of matches, containing the whole cache
        # document as its first item. Avoid the upstream legacy '.' read too.
        matches = self.redis.json().get(self._key(prompt, llm_string), "$")
        result = None
        if matches:
            cached = matches[0]
            # Preserve compatibility with both upstream multi-generation
            # documents and older single-generation documents.
            values = cached if isinstance(cached, list) else [cached]
            result = [
                loads(json.dumps(value), allowed_objects="core")
                for value in values
            ]
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
        # langchain-redis 0.2.6 uses the legacy '.' root, which some hosted
        # Redis implementations reject when creating a new JSON document.
        # Preserve its key/serialization format while using JSONPath '$'.
        key = self._key(prompt, llm_string)
        json_value = [json.loads(dumps(generation)) for generation in return_val]
        self.redis.json().set(key, "$", json_value)
        if self.ttl is not None:
            self.redis.expire(key, self.ttl)
