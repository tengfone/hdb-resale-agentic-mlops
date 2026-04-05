import importlib
import os
import tempfile
import unittest
from pathlib import Path


class TestEnvLoading(unittest.TestCase):
    def setUp(self):
        self._cwd = Path.cwd()
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.chdir(self._cwd)
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_load_repo_env_reads_cwd_dotenv_without_overriding_existing_env(self):
        from hdb_resale_mlops import env as env_module

        env_module._LOADED_ENV_FILES.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            os.chdir(tmp_path)
            (tmp_path / ".env").write_text(
                "\n".join(
                    [
                        "OPENAI_BASE_URL=https://proxy.example.com/v1",
                        "MAESTRO_HTTP_PROXY=http://maestro-proxy:8080",
                        "MAESTRO_HTTPS_PROXY=http://maestro-proxy:8443",
                        'export OPENAI_MODEL="gpt-5-nano"',
                        "MARKET_RESEARCH_PROVIDER=openai # inline comment",
                        "MODEL_REVIEWER='alice'",
                    ]
                )
            )
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("MARKET_RESEARCH_PROVIDER", None)
            os.environ.pop("MODEL_REVIEWER", None)
            os.environ.pop("MAESTRO_HTTP_PROXY", None)
            os.environ.pop("MAESTRO_HTTPS_PROXY", None)
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            os.environ["OPENAI_MODEL"] = "preset-model"

            loaded = env_module.load_repo_env()

            self.assertEqual(loaded, (tmp_path / ".env").resolve())
            self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://proxy.example.com/v1")
            self.assertEqual(os.environ["OPENAI_MODEL"], "preset-model")
            self.assertEqual(os.environ["MARKET_RESEARCH_PROVIDER"], "openai")
            self.assertEqual(os.environ["MODEL_REVIEWER"], "alice")
            self.assertEqual(
                os.environ["MAESTRO_HTTP_PROXY"], "http://maestro-proxy:8080"
            )
            self.assertEqual(
                os.environ["MAESTRO_HTTPS_PROXY"], "http://maestro-proxy:8443"
            )
            self.assertIsNone(os.environ.get("HTTP_PROXY"))
            self.assertIsNone(os.environ.get("HTTPS_PROXY"))

    def test_package_import_autoloads_cwd_dotenv(self):
        import hdb_resale_mlops
        from hdb_resale_mlops import env as env_module

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            os.chdir(tmp_path)
            (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-notebook-test\n")
            os.environ.pop("OPENAI_API_KEY", None)
            env_module._LOADED_ENV_FILES.clear()

            importlib.reload(hdb_resale_mlops)

            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "sk-notebook-test")

    def test_maestro_proxy_env_temporarily_sets_standard_proxy_vars(self):
        from hdb_resale_mlops.env import maestro_proxy_env

        os.environ["MAESTRO_HTTP_PROXY"] = "http://maestro-proxy:8080"
        os.environ["MAESTRO_HTTPS_PROXY"] = "http://maestro-proxy:8443"
        os.environ["HTTP_PROXY"] = "http://existing-proxy:3128"
        os.environ["HTTPS_PROXY"] = "http://existing-proxy:4443"

        with maestro_proxy_env():
            self.assertEqual(os.environ["HTTP_PROXY"], "http://maestro-proxy:8080")
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://maestro-proxy:8443")
            self.assertEqual(os.environ["http_proxy"], "http://maestro-proxy:8080")
            self.assertEqual(os.environ["https_proxy"], "http://maestro-proxy:8443")

        self.assertEqual(os.environ["HTTP_PROXY"], "http://existing-proxy:3128")
        self.assertEqual(os.environ["HTTPS_PROXY"], "http://existing-proxy:4443")
        self.assertIsNone(os.environ.get("http_proxy"))
        self.assertIsNone(os.environ.get("https_proxy"))

    def test_collect_sagemaker_forwarded_env_includes_shared_openai_settings(self):
        from hdb_resale_mlops.env import collect_sagemaker_forwarded_env

        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)
        os.environ.pop("MAESTRO_HTTP_PROXY", None)
        os.environ.pop("MAESTRO_HTTPS_PROXY", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ["OPENAI_BASE_URL"] = "https://proxy.example.com/v1"
        os.environ["OPENAI_MODEL"] = "gpt-5-nano"
        os.environ["OPENAI_JUDGE_MODEL"] = "gpt-5-mini"
        os.environ["ENABLE_JUDGE_EVAL"] = "true"
        os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.example.com"
        os.environ["MAESTRO_HTTP_PROXY"] = "http://maestro-proxy:8080"
        os.environ["MAESTRO_HTTPS_PROXY"] = "http://maestro-proxy:8443"

        forwarded = collect_sagemaker_forwarded_env()

        self.assertEqual(forwarded["OPENAI_BASE_URL"], "https://proxy.example.com/v1")
        self.assertEqual(forwarded["OPENAI_API_BASE"], "https://proxy.example.com/v1")
        self.assertEqual(forwarded["OPENAI_MODEL"], "gpt-5-nano")
        self.assertEqual(forwarded["OPENAI_JUDGE_MODEL"], "gpt-5-mini")
        self.assertEqual(forwarded["ENABLE_JUDGE_EVAL"], "true")
        self.assertEqual(forwarded["MLFLOW_TRACKING_URI"], "https://mlflow.example.com")
        self.assertEqual(
            forwarded["MAESTRO_HTTP_PROXY"], "http://maestro-proxy:8080"
        )
        self.assertEqual(
            forwarded["MAESTRO_HTTPS_PROXY"], "http://maestro-proxy:8443"
        )
        self.assertNotIn("HTTP_PROXY", forwarded)
        self.assertNotIn("HTTPS_PROXY", forwarded)


if __name__ == "__main__":
    unittest.main()
