import ast
import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("state_compare", HERE/"checkpoint_state_compare.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


@pytest.fixture
def states():
    expected = {"seed":401, "cell":"HL", "run_id":"seed401-owned-run",
                "source_revision":"a"*64,"data_manifest_hash":"b"*64,"eval_revision":"c"*64}
    config = {"seed":401,"cell":"HL"}
    order = [0,1,2,3]
    order_hash = compare.hashlib.sha256(json.dumps(order,separators=(",",":")).encode()).hexdigest()
    left = {
        "schema_version":1,"execution_device":"cpu","checkpoint_id":"one",
        "latest_verified_checkpoint_id":"one","last_finite_checkpoint_id":"one",
        "run_id":expected["run_id"],"config":config,"canonical_config_json":json.dumps(config),
        "config_hash":"d"*64,"training_config":{"optimizer":"adamw"},"training_config_hash":"e"*64,
        "source_revision":"a"*64,"data_manifest_hash":"b"*64,"tokenizer_revision":"fixed-tokenizer",
        "eval_revision":"c"*64,"protocol_id":"colab-free-mini-stability-v1","source_kind":"measured",
        "purpose":"mini_exploration_only","model_state_dict":{"layer.weight":torch.tensor([1.,2.])},
        "optimizer_state_dict":{"state":{0:{"step":torch.tensor(1.),
            "exp_avg":torch.tensor([0.1]),"exp_avg_sq":torch.tensor([0.01])}},
            "param_groups":[{"params":[0],"lr":0.0003}]},
        "last_step_diagnostics":{"finite":True},"scheduler_state_dict":{"tokens_seen":4096},
        "scaler_state":None,"rng_state":{"python":random.getstate(),"numpy":np.random.get_state(),
            "torch_cpu":torch.get_rng_state(),"used_device_rng_states":{"cpu":torch.get_rng_state()}},
        "data_order":order,"data_order_hash":order_hash,"epoch":0,"cursor":8,
        "trained_target_tokens":4096,"optimizer_step":1,
    }
    return left,copy.deepcopy(left),expected


def test_keys_match_actual_trainer_payload():
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root/"src/train.py").read_text())
    fn = next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=="_checkpoint_payload")
    value = next(node.value for node in fn.body if isinstance(node,ast.Return))
    assert set(compare.FIELDS) == {key.value for key in value.keys}


def test_exact_tensors_arrays_scalars_pass(states):
    report = compare.compare_payloads(*states)
    assert report["status"] == "exact-state-verified"
    assert report["excluded_fields"] == []
    assert len(report["checked_fields"]) == 30


@pytest.mark.parametrize("field",compare.FIELDS)
def test_missing_required_key_refused_even_if_both_omit(states,field):
    left,right,expected=states
    del left[field]
    del right[field]
    with pytest.raises(compare.StateMismatch,match="required_fields"):
        compare.compare_payloads(left,right,expected)


@pytest.mark.parametrize("kind",["rng","optimizer","order","model","dtype","scheduler","run","seed","source","eval","device","scaler"])
def test_state_mismatch_refused_without_values(states,kind):
    left,right,expected=states
    if kind=="rng":
        right["rng_state"]["torch_cpu"][0] ^= 1
    elif kind=="optimizer":
        right["optimizer_state_dict"]["state"][0]["exp_avg"][0] += 1
    elif kind=="order":
        right["data_order"][0],right["data_order"][1]=right["data_order"][1],right["data_order"][0]
    elif kind=="model":
        right["model_state_dict"]["layer.weight"][0] = 12345.678
    elif kind=="dtype":
        right["model_state_dict"]["layer.weight"] = right["model_state_dict"]["layer.weight"].double()
    elif kind=="scheduler":
        right["scheduler_state_dict"]["tokens_seen"] += 1
    elif kind=="run":
        right["run_id"]="foreign"
    elif kind=="seed":
        right["config"]["seed"]=17
    elif kind=="source":
        right["source_revision"]="f"*64
    elif kind=="eval":
        right["eval_revision"]="f"*64
    elif kind=="device":
        right["execution_device"]="cuda"
    elif kind=="scaler":
        right["scaler_state"]={}
    with pytest.raises(compare.StateMismatch) as exc:
        compare.compare_payloads(left,right,expected)
    assert "12345" not in str(exc.value) and "foreign" not in str(exc.value)


@pytest.mark.parametrize("field",["torch_cpu","numpy","python","used_device_rng_states"])
def test_rng_omission_refused_in_both(states,field):
    left,right,expected=states
    del left["rng_state"][field]
    del right["rng_state"][field]
    with pytest.raises(compare.StateMismatch,match="rng_state.required_fields"):
        compare.compare_payloads(left,right,expected)


def test_optimizer_moment_omission_refused_in_both(states):
    left,right,expected=states
    del left["optimizer_state_dict"]["state"][0]["exp_avg"]
    del right["optimizer_state_dict"]["state"][0]["exp_avg"]
    with pytest.raises(compare.StateMismatch,match="required_fields"):
        compare.compare_payloads(left,right,expected)


def test_continuation_excludes_only_generated_uuids(states):
    left,right,expected=states
    for field in compare.UUID_FIELDS:
        right[field]="two"
    with pytest.raises(compare.StateMismatch,match="checkpoint_id"):
        compare.compare_payloads(left,right,expected)
    report=compare.compare_payloads(left,right,expected,mode="continuation")
    assert set(report["excluded_fields"])==compare.UUID_FIELDS
    right["optimizer_step"]=2
    right["cursor"]=16
    right["trained_target_tokens"]=8192
    right["scheduler_state_dict"]["tokens_seen"]=8192
    with pytest.raises(compare.StateMismatch):
        compare.compare_payloads(left,right,expected,mode="continuation")


def test_verified_files_and_cli(states,tmp_path):
    import hashlib
    left,right,expected=states
    paths=[tmp_path/"left.pt",tmp_path/"right.pt"]
    for p,payload in zip(paths,(left,right)):
        torch.save(payload,p)
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    identity=tmp_path/"identity.json"
    identity.write_text(json.dumps(expected))
    report=compare.compare_files(paths[0],hashes[0],paths[1],hashes[1],expected)
    assert report["status"]=="exact-state-verified"
    command=[sys.executable,str(HERE/"checkpoint_state_compare.py"),"--left",str(paths[0]),
        "--left-sha256",hashes[0],"--right",str(paths[1]),"--right-sha256",hashes[1],
        "--expected-identity",str(identity)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["status"]=="exact-state-verified"
    with pytest.raises(compare.StateMismatch,match="sha256"):
        compare.compare_files(paths[0],"0"*64,paths[1],hashes[1],expected)
