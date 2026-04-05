import os
import unittest

from hdb_resale_mlops.config import RuntimeConfig


class TestRuntimeConfig(unittest.TestCase):
    def setUp(self):
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_runtime_config_uses_maestro_proxy_vars_for_data_gov_requests(self):
        os.environ["MAESTRO_HTTP_PROXY"] = "http://maestro-proxy:8080"
        os.environ["MAESTRO_HTTPS_PROXY"] = "http://maestro-proxy:8443"

        runtime_config = RuntimeConfig.from_env()

        self.assertEqual(
            runtime_config.maestro_proxies,
            {
                "http": "http://maestro-proxy:8080",
                "https": "http://maestro-proxy:8443",
            },
        )


if __name__ == "__main__":
    unittest.main()
