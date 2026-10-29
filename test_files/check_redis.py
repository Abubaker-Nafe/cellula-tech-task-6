import redis

client = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

print("Connected:", client.ping())

client.set("task6:check", "Redis is working", ex=60)
print("Stored value:", client.get("task6:check"))

client.close()