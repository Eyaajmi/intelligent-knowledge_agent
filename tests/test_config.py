import os

from app.config import Settings, load_env_file


def test_env_file_is_loaded_without_overriding_existing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# commentaire\nCHAT_MODEL=llama3.2:3b\nQUERY_PREFIX=\"search_query: \"\n\nTOP_K=7\nDEJA=depuis_le_fichier\n",
        encoding="utf-8",
    )
    for key in ("CHAT_MODEL", "QUERY_PREFIX", "TOP_K"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEJA", "depuis_l_environnement")
    load_env_file(env)
    s = Settings.from_env()
    assert s.chat_model == "llama3.2:3b" and s.top_k == 7
    assert s.query_prefix == "search_query: "  # espace final conservé grâce aux guillemets
    assert os.environ["DEJA"] == "depuis_l_environnement"
    for key in ("CHAT_MODEL", "QUERY_PREFIX", "TOP_K"):
        monkeypatch.delenv(key, raising=False)


def test_missing_env_file_is_ignored(tmp_path):
    load_env_file(tmp_path / "absent.env")


def test_cors_origins_list(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    assert Settings.from_env().cors_origins == ["http://a.test", "http://b.test"]
