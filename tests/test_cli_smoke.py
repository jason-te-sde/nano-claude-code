from nanoclaude.cli.main import main


def test_version_flag_prints_and_exits_zero(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "nano-claude-code" in out


def test_no_arguments_start_the_repl_and_with_no_configuration_point_at_init(
    tmp_path, monkeypatch, capsys
):
    # In a home and a directory of its own: what ncc finds there is the whole configuration,
    # and nothing of whoever runs the tests.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    code = main([])
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == ""
    assert "ncc init" in captured.err
