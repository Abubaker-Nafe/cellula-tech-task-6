"""Offline checks for local/cloud endpoint selection and the live diagnostic."""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from redis.connection import SSLConnection
from redis.exceptions import ResponseError

import redis_config
from test_files import check_redis


class RedisConfigurationTests(unittest.TestCase):
    def test_missing_setting_keeps_local_docker_working(self):
        with patch.dict(os.environ):
            os.environ.pop("REDIS_URL", None)
            with redis_config.create_redis_client() as client:
                options = client.connection_pool.connection_kwargs
                self.assertEqual(options["host"], "127.0.0.1")
                self.assertEqual(options["port"], 6379)
                self.assertEqual(options["db"], 0)

    def test_hosted_url_preserves_tls_authentication_and_encoded_password(self):
        with patch.dict(os.environ, {
            "REDIS_URL": "rediss://default:fake%40pass%3Aword@redis.example.invalid:16443/0",
        }):
            with redis_config.create_redis_client() as client:
                self.assertIs(client.connection_pool.connection_class, SSLConnection)
                options = client.connection_pool.connection_kwargs
                self.assertEqual(options["host"], "redis.example.invalid")
                self.assertEqual(options["port"], 16443)
                self.assertEqual(options["username"], "default")
                self.assertEqual(options["password"], "fake@pass:word")
                self.assertEqual(options["socket_connect_timeout"], 5)
                self.assertEqual(options["socket_timeout"], 5)

    def test_invalid_url_errors_do_not_disclose_credentials(self):
        for url in ("", "https://default:fake-secret@host", "redis://", "redis://default:fake-secret@host:bad"):
            with self.subTest(url=url), patch.dict(os.environ, {"REDIS_URL": url}):
                with self.assertRaises(ValueError) as raised:
                    redis_config.get_redis_url()
                self.assertIn("REDIS_URL", str(raised.exception))
                self.assertNotIn("fake-secret", str(raised.exception))

    def test_diagnostic_checks_json_and_removes_only_its_own_probe(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.ping.return_value = True
        client.json.return_value.get.return_value = {"message": "Redis is working"}
        output = io.StringIO()
        with patch.object(check_redis, "create_redis_client", return_value=client):
            with redirect_stdout(output):
                check_redis.main()
        key = client.json.return_value.set.call_args.args[0]
        self.assertTrue(key.startswith("task6:check:"))
        client.expire.assert_called_once_with(key, 60)
        client.delete.assert_called_once_with(key)
        client.flushdb.assert_not_called()
        self.assertIn("Redis JSON support: True", output.getvalue())

    def test_ping_alone_does_not_pass_without_json_support(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.ping.return_value = True
        client.json.return_value.set.side_effect = ResponseError("unknown command JSON.SET")
        output = io.StringIO()
        with patch.object(check_redis, "create_redis_client", return_value=client):
            with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                check_redis.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertNotIn("Redis JSON support: True", output.getvalue())
        self.assertIn("Redis check failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
