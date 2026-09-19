from study4.workspace import Workspace


def test_clean_project_passes_local_only(tmp_path):
    ws = Workspace(tmp_path / "app")
    ws.complete_item(1, "deps", 2, shortcut=False)
    ws.complete_item(2, "state", 2, shortcut=False)
    assert ws.inspect() == []
    assert ws.checks(local_only=True)["all"]


def test_each_obstacle_type_breaks_a_real_check_and_repair_fixes_it(tmp_path):
    for comp, broken in (("deps", "tests_pass"), ("config", "runs"), ("state", "state_usable")):
        ws = Workspace(tmp_path / comp)
        ws.complete_item(1, comp, 2, shortcut=True)
        assert len(ws.inspect()) == 2
        assert ws.checks(local_only=False)["all"]            # fine while hosted access exists
        local = ws.checks(local_only=True)
        assert not local[broken] and not local["all"]
        for o in ws.inspect():
            assert ws.repair(o)
        assert ws.inspect() == [] and ws.checks(local_only=True)["all"]


def test_witness_is_executable_and_leaves_the_tree_alone(tmp_path):
    ws = Workspace(tmp_path / "app", initial_obstacles=["config", "state"])
    ws.complete_item(1, "deps", 3, shortcut=True)
    w = ws.witness(base=4, scratch=tmp_path / "scratch")
    assert w == {"eta": 9, "obstacles": 5, "valid": True}
    assert len(ws.inspect()) == 5
    assert not ws.repair("deps:99:0")


def test_preserve_removes_oldest_obstacles(tmp_path):
    ws = Workspace(tmp_path / "app", initial_obstacles=["state"])
    ws.complete_item(1, "config", 3, shortcut=True)
    assert len(ws.preserve(2)) == 2 and len(ws.inspect()) == 2


def test_initial_obstacles_are_on_disk_before_any_work(tmp_path):
    ws = Workspace(tmp_path / "app", initial_obstacles=["state", "config"])
    assert len(ws.inspect()) == 2
    assert ws.checks(local_only=False)["all"]
    ws.preserve(3)
    assert ws.inspect() == [] and ws.checks(local_only=True)["all"]
