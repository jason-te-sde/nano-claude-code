from nanoclaude.cli.main import main


def test_version_flag_prints_and_exits_zero(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "nano-claude-code" in out


def test_no_arguments_is_an_error_with_guidance(capsys):
    code = main([])
    _ = capsys.readouterr().out + capsys.readouterr().err
    assert code == 2
