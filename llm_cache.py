from langchain_redis import RedisCache


class LoggedRedisCache(RedisCache):
    def lookup(self, prompt, llm_string):
        result = super().lookup(prompt, llm_string)
        print("Cache:", "HIT" if result is not None else "MISS")
        return result