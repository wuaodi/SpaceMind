r"""
UE 编辑器 Python 脚本：在场景中放置目标卫星。

参考:
- D:/project/mm-Space-Bench/data_collect/ue_satellite_swap.py

使用方式（UE -> Output Log -> Python）:
import sys, importlib
sys.path.append(r"D:\project\SpaceMind\SpaceMind_UE5\environments\satellite_pipeline")
import ue_place_satellite as sat_tool
importlib.reload(sat_tool)
sat_tool.place_selected_satellite()

你通常只需要改下面两个配置:
1. SELECTED_SATELLITE
2. ASSET_ROOT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import unreal


# -----------------------------
# UE 场景配置
# -----------------------------
MESH_ROOT = "/Game/Meshes/Spacecraft_136"
ASSET_ROOT = MESH_ROOT
SATELLITE_DESCRIPTION_PATH = Path(__file__).with_name("satellite_descriptions.json")
TARGET_SATELLITE_NAMES = [
    "CAPSTONE",
    "IBEX",
    "BioSentinel",
    "New_Horizons",
    "Huygens",
]
SELECTED_SATELLITE = os.getenv("SPACEMIND_TARGET_SATELLITE", "Huygens")

BASE_LOCATION = (0.0, 0.0, 707099978.463)
BASE_ROTATION = (0.0, 0.0, 0.0)
BASE_SCALE = (1.0, 1.0, 1.0)
ACTOR_FOLDER_ROOT = "Satellite"
CLEAR_OLD_SATELLITES = True


def _log(msg: str) -> None:
    try:
        unreal.log(msg)
    except Exception:
        pass


def _as_vector(loc: Sequence[float]) -> unreal.Vector:
    return unreal.Vector(float(loc[0]), float(loc[1]), float(loc[2]))


def _as_rotator(rot: Sequence[float]) -> unreal.Rotator:
    return unreal.Rotator(float(rot[0]), float(rot[1]), float(rot[2]))


def _load_satellite_descriptions() -> list[dict]:
    with SATELLITE_DESCRIPTION_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    satellites = data.get("satellites")
    if not isinstance(satellites, list):
        raise ValueError(f"Invalid satellite description file: {SATELLITE_DESCRIPTION_PATH}")
    return satellites


def _build_target_satellite_options() -> dict[str, str]:
    satellites = _load_satellite_descriptions()
    by_name = {str(item.get("name")): item for item in satellites if item.get("name")}

    options: dict[str, str] = {}
    for name in TARGET_SATELLITE_NAMES:
        item = by_name.get(name)
        if not item:
            raise ValueError(f"Satellite '{name}' was not found in {SATELLITE_DESCRIPTION_PATH}")

        diameter = float(item.get("max_diameter_meters", -1.0))
        if not (0.0 <= diameter <= 3.0):
            raise ValueError(
                f"Satellite '{name}' diameter {diameter}m is outside the required 0-3m range."
            )
        options[name] = name
    return options


SATELLITE_OPTIONS = _build_target_satellite_options()


def _asset_folder_for_satellite(name: str) -> str:
    if name not in SATELLITE_OPTIONS:
        raise ValueError(
            f"Unsupported satellite '{name}'. Available: {', '.join(SATELLITE_OPTIONS.keys())}"
        )
    sub = SATELLITE_OPTIONS[name]
    base = ASSET_ROOT.rstrip("/")
    return f"{base}/{sub}" if sub else base


def list_level_actors() -> list[unreal.Actor]:
    try:
        return list(unreal.EditorLevelLibrary.get_all_level_actors())
    except Exception:
        sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return list(sub.get_all_level_actors())


def delete_actors_in_folder_root(folder_root: str) -> int:
    folder_root = folder_root.strip("/")
    if not folder_root:
        return 0

    to_delete: list[unreal.Actor] = []
    for actor in list_level_actors():
        try:
            actor_folder = str(actor.get_folder_path()).strip("/")
        except Exception:
            continue
        if actor_folder == folder_root or actor_folder.startswith(folder_root + "/"):
            to_delete.append(actor)

    if not to_delete:
        return 0

    try:
        for actor in to_delete:
            unreal.EditorLevelLibrary.destroy_actor(actor)
    except Exception:
        sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for actor in to_delete:
            sub.destroy_actor(actor)

    _log(f"Deleted {len(to_delete)} actors in folder root '{folder_root}'")
    return len(to_delete)


def list_static_mesh_assets_in_folder(folder: str) -> list[str]:
    folder = folder.rstrip("/")
    assets = unreal.EditorAssetLibrary.list_assets(folder, recursive=True, include_folder=False)
    mesh_assets: list[str] = []
    for path in assets:
        obj = unreal.EditorAssetLibrary.load_asset(path)
        if isinstance(obj, unreal.StaticMesh):
            mesh_assets.append(path)
    return mesh_assets


def spawn_static_mesh_actor(
    mesh: unreal.StaticMesh,
    location: unreal.Vector,
    rotation: unreal.Rotator,
    scale: Sequence[float],
    actor_label: str,
    actor_folder: str,
):
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, location, rotation)
    smc = actor.static_mesh_component
    smc.set_static_mesh(mesh)
    try:
        smc.set_mobility(unreal.ComponentMobility.STATIC)
    except Exception:
        pass

    try:
        actor.set_actor_scale3d(unreal.Vector(float(scale[0]), float(scale[1]), float(scale[2])))
    except Exception:
        pass

    try:
        actor.set_actor_label(actor_label, mark_dirty=True)
    except Exception:
        actor.set_actor_label(actor_label)

    try:
        actor.set_folder_path(actor_folder)
    except Exception:
        pass

    return actor


def place_selected_satellite(satellite_name: str | None = None) -> int:
    satellite = satellite_name or SELECTED_SATELLITE
    mesh_folder = _asset_folder_for_satellite(satellite)
    actor_folder = f"{ACTOR_FOLDER_ROOT.rstrip('/')}/{satellite}"

    if CLEAR_OLD_SATELLITES:
        delete_actors_in_folder_root(ACTOR_FOLDER_ROOT)

    mesh_paths = list_static_mesh_assets_in_folder(mesh_folder)
    if not mesh_paths:
        raise RuntimeError(
            f"No StaticMesh assets found in '{mesh_folder}'. "
            f"Please check whether the FBX has been imported into UE content browser."
        )

    base_location = _as_vector(BASE_LOCATION)
    base_rotation = _as_rotator(BASE_ROTATION)
    spawned = 0

    for path in mesh_paths:
        mesh = unreal.EditorAssetLibrary.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            continue

        asset_name = unreal.Paths.get_base_filename(path)
        actor_label = f"{satellite}_{asset_name}"
        spawn_static_mesh_actor(
            mesh=mesh,
            location=base_location,
            rotation=base_rotation,
            scale=BASE_SCALE,
            actor_label=actor_label,
            actor_folder=actor_folder,
        )
        spawned += 1

    _log(
        f"Placed satellite '{satellite}' from '{mesh_folder}', "
        f"spawned={spawned}, location={BASE_LOCATION}, rotation={BASE_ROTATION}, scale={BASE_SCALE}"
    )
    return spawned


def list_target_satellites() -> list[str]:
    return list(SATELLITE_OPTIONS.keys())


def dump_name_label_mapping(out_path: str, only_static_mesh_actor: bool = True) -> int:
    lines: list[str] = []
    for actor in list_level_actors():
        try:
            if only_static_mesh_actor and actor.get_class().get_name() != "StaticMeshActor":
                continue
            lines.append(f"{actor.get_name()}|{actor.get_actor_label()}")
        except Exception:
            continue

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    _log(f"dump_name_label_mapping: wrote {len(lines)} lines -> {out_path}")
    return len(lines)
