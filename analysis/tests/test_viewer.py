import tempfile
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import viewer


def test_config_kind_and_expected_files():
    cfg = {"run_dir": "/data/HFSA", "init": 2024092400,
           "out_dir": "analysis/output/helene"}
    assert viewer.config_kind(cfg) == "case"
    names = [p.name for p in viewer.expected_case_files(Path("helene.yaml"), cfg)]
    assert names == [
        "parent_qpf_helene_2024092400.png",
        "ets_full_helene_2024092400.png",
        "rmse_scatter_helene_2024092400.png",
    ]


def test_config_kind_recognizes_paper_configs():
    assert viewer.config_kind({"models": {}, "best_track": "/best"}) == "paper"
    assert viewer.config_kind({"storms": ["alpha.yaml"]}) == "paper_suite"
    assert viewer.model_label({"models": {"HAFS-A": "/a", "HAFS-B": "/b"}}) == \
        "HAFS-A / HAFS-B"


def test_manifest_groups_case_files_and_orphans():
    with tempfile.TemporaryDirectory(dir=viewer.REPO_ROOT) as tmp:
        root = Path(tmp)
        storms = root / "storms"
        output = root / "output"
        case_out = output / "alpha"
        storms.mkdir()
        case_out.mkdir(parents=True)
        cfg_path = storms / "alpha_hfsa.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "run_dir": "/data/alpha/HFSA", "storm_name": "Alpha",
            "init": 2025010100, "out_dir": str(case_out),
        }))
        (case_out / "ets_full_alpha_hfsa_2025010100.png").write_bytes(b"png")
        (output / "loose.png").write_bytes(b"png")

        manifest = viewer.build_manifest([cfg_path], output)
        case = manifest["cases"][0]
        assert case["storm"] == "Alpha"
        assert case["model"] == "HAFS-A"
        assert case["files"][0]["name"].startswith("ets_full_")
        assert manifest["cases"][1]["id"] == "other_graphics"


def test_discover_configs_filters_by_stem():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "one.yaml").write_text("run_dir: /one\n")
        (root / "two.yaml").write_text("run_dir: /two\n")
        assert [p.stem for p in viewer.discover_configs(root, ["two"])] == ["two"]


def test_remote_access_instructions_use_ssh_tunnel():
    lines = viewer.access_instructions(
        "127.0.0.1", 8765, "hercules",
        environ={"SSH_CONNECTION": "192.0.2.1 50000 192.0.2.2 22"},
    )
    assert "remote login node" in lines[0]
    assert "ssh -N -L 8765:127.0.0.1:8765 hercules" in lines[2]
    assert "  http://127.0.0.1:8765" in lines
    assert any("--export" in line for line in lines)


def test_local_access_instructions_need_no_tunnel():
    lines = viewer.access_instructions("127.0.0.1", 8765, environ={})
    assert lines == ["Viewer: http://127.0.0.1:8765  (Ctrl-C to stop)"]


def test_offline_export_embeds_graphics():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "plot.png").write_bytes(b"fake png")
        manifest = {"updated": "2026-01-01T00:00:00", "cases": [{
            "id": "alpha", "storm": "Alpha", "model": "HAFS-A",
            "init": "2025010100", "kind": "case", "config": "alpha.yaml",
            "files": [{"name": "plot.png", "url": "plot.png",
                       "modified": "2026-01-01T00:00:00", "size": 8}],
        }]}
        destination = root / "viewer.html"
        viewer.export_offline_html(manifest, root, destination)
        html = destination.read_text()
        assert "data:image/png;base64," in html
        assert "Alpha" in html
        assert "refresh();setInterval" not in html
