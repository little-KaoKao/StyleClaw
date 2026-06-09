from styleclaw.web.launch import build_server_config


def test_build_server_config_defaults():
    cfg = build_server_config(port=8800, open_browser=False)
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 8800


def test_web_command_registered():
    from typer.testing import CliRunner
    from styleclaw.cli import app

    result = CliRunner().invoke(app, ["web", "--help"])
    assert result.exit_code == 0
    assert "port" in result.output.lower()
