"""POVBench observer-relative spatial grounding integration.

The prompt, response parser, geometry, and aggregation below follow the
POVBench reference implementation at revision
``fbaf9d7abdbeb1c19f98fce3076d3ad51f14e8b1``.  The upstream implementation
is Copyright (c) 2026 Mimo Shirasaka and licensed under the MIT License.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load

from .image_base import ImageBaseDataset


SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512
POVBENCH_REPO = 'owl-owl/POVBench'
POVBENCH_REVISION = 'fbaf9d7abdbeb1c19f98fce3076d3ad51f14e8b1'
CONDITION_NAMES = {'a': 'Inferred', 'b': 'Stated', 'c': 'Given'}


def _coord_description(width: int, height: int) -> str:
    return (
        'Image coordinate system:\n'
        '  - Origin [x=0, y=0] is the TOP-LEFT corner of the image.\n'
        f'  - x-axis points RIGHT: x=0 is the left edge, x={width - 1} is the right edge.\n'
        f'  - y-axis points DOWN:  y=0 is the top edge,  y={height - 1} is the bottom edge.\n'
        f'  - Image size: {width} x {height} pixels.\n'
        f'  - Valid range: x ∈ [0, {width - 1}], y ∈ [0, {height - 1}].'
    )


def prompt_ab(sentence: str, n_images: int, width: int = SCREEN_WIDTH,
              height: int = SCREEN_HEIGHT) -> str:
    return (
        "You are the intelligent brain system of a home-assistance robot. "
        "You are given first-person perspective images captured during the robot's exploration of a home.\n\n"
        f'The {n_images} images above (labeled Image 0 through Image {n_images - 1}) were all taken '
        'during exploration of the same single-story house. '
        'Each image is preceded by its label ("Image 0:", "Image 1:", etc.).\n\n'
        f'Sentence: "{sentence}"\n\n'
        'The resident has asked the robot to retrieve an object and is describing where they last saw it.'
        "Your mission is to predict, as accurately as possible, where that object is located, based on the resident's description in the sentence above.\n\n"
        'Task:\n'
        f'1. Select the image (0 to {n_images - 1}) that best represents the scene described in the sentence '
        "— the image where both the observer's furniture and the reference landmark are most clearly visible.\n"
        '2. In the selected image, predict the center of the target object as pixel coordinates.\n'
        '   The target object may not actually be visible in the image. '
        'Based on the sentence, predict as faithfully and accurately as possible '
        'where the center of the object WOULD be located.\n\n'
        f'{_coord_description(width, height)}\n\n'
        'IMPORTANT: When predicting coordinates, pay close attention to the origin position and '
        'axis directions, and ensure all values fall within the valid range.\n\n'
        'Output ONLY valid JSON, no other text:\n'
        f'{{"selected_image": <integer 0-{n_images - 1}>, "x": <integer 0-{width - 1}>, "y": <integer 0-{height - 1}>}}'
    )


def prompt_c(sentence: str, width: int = SCREEN_WIDTH,
             height: int = SCREEN_HEIGHT) -> str:
    return (
        'You are the intelligent brain system of a home-assistance robot.\n\n'
        f'Sentence: "{sentence}"\n\n'
        'The resident has asked the robot to retrieve an object and is describing where they last saw it. '
        'Your mission is to predict, as accurately as possible, where that object is located — '
        "based on the resident's description in the sentence above.\n\n"
        'The image provided shows the target scene from the perspective of the resident '
        'at the moment they spotted the target object.\n\n'
        'Task: Based on the sentence, predict the center of the target object as pixel coordinates.\n'
        'The target object may not actually be visible in the image. '
        'Based on the sentence, predict as faithfully and accurately as possible '
        'where the center of the object WOULD be located.\n\n'
        f'{_coord_description(width, height)}\n\n'
        'IMPORTANT: When predicting coordinates, pay close attention to the origin position and '
        'axis directions, and ensure all values fall within the valid range.\n\n'
        'Output ONLY valid JSON, no other text:\n'
        f'{{"x": <integer 0-{width - 1}>, "y": <integer 0-{height - 1}>}}'
    )


def extract_all_jsons(text: str) -> list[dict]:
    """Extract non-nested JSON objects in their source order."""
    results = []
    for match in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            results.append(json.loads(match.group()))
        except (ValueError, TypeError):
            pass
    return results


def _best_coord_json(candidates: list[dict], keys: tuple[str, ...],
                     width: int = SCREEN_WIDTH,
                     height: int = SCREEN_HEIGHT) -> dict | None:
    valid_any = None
    for obj in reversed(candidates):
        if not all(key in obj for key in keys):
            continue
        try:
            x, y = float(obj['x']), float(obj['y'])
        except (KeyError, ValueError, TypeError):
            continue
        if valid_any is None:
            valid_any = obj
        if 0 <= x <= width - 1 and 0 <= y <= height - 1:
            return obj
    return valid_any


def parse_response(text: str, condition: str, width: int = SCREEN_WIDTH,
                   height: int = SCREEN_HEIGHT) -> dict | None:
    keys = ('selected_image', 'x', 'y') if condition in ('a', 'b') else ('x', 'y')
    obj = _best_coord_json(extract_all_jsons(str(text).strip()), keys, width, height)
    if obj is None:
        return None
    try:
        answer = {'x': float(obj['x']), 'y': float(obj['y'])}
        if condition in ('a', 'b'):
            answer['selected_image'] = int(obj['selected_image'])
        return answer
    except (KeyError, ValueError, TypeError):
        return None


def _camera_frame(yaw_degrees: float, horizon_degrees: float):
    yaw, horizon = math.radians(yaw_degrees), math.radians(horizon_degrees)
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    forward = (
        math.sin(yaw) * math.cos(horizon),
        -math.sin(horizon),
        math.cos(yaw) * math.cos(horizon),
    )
    up = (
        math.sin(yaw) * math.sin(horizon),
        math.cos(horizon),
        math.cos(yaw) * math.sin(horizon),
    )
    return right, up, forward


def project_3d_to_2d(target_pos, cam_pos, cam_yaw_deg: float,
                     cam_horizon_deg: float, fov_deg: float = 100.0,
                     screen_width: int = 512, screen_height: int = 512):
    if isinstance(target_pos, dict):
        tx, ty, tz = target_pos['x'], target_pos['y'], target_pos['z']
    else:
        tx, ty, tz = map(float, target_pos)
    cx, cy, cz = map(float, cam_pos)
    dx, dy, dz = tx - cx, ty - cy, tz - cz
    right, up, forward = _camera_frame(cam_yaw_deg, cam_horizon_deg)
    x_cam = dx * right[0] + dy * right[1] + dz * right[2]
    y_cam = dx * up[0] + dy * up[1] + dz * up[2]
    z_cam = dx * forward[0] + dy * forward[1] + dz * forward[2]
    if z_cam <= 1e-6:
        return 0.5, 0.5, False
    focal = (screen_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    x_norm = (focal * x_cam / z_cam + screen_width / 2.0) / screen_width
    y_norm = (screen_height / 2.0 - focal * y_cam / z_cam) / screen_height
    in_frame = 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0
    return float(np.clip(x_norm, 0.0, 1.0)), float(np.clip(y_norm, 0.0, 1.0)), in_frame


_GROUP_SPECIAL = {'Bed', 'CounterTop', 'Desk', 'Desktop', 'DiningTable', 'Dresser', 'SideTable'}
_ANCHOR_SURFACE_KEYWORDS = {
    'Bed': ['bed'], 'CounterTop': ['counter', 'countertop'], 'Desk': ['desk'],
    'Desktop': ['desktop'], 'DiningTable': ['dining table'], 'Dresser': ['dresser'],
    'SideTable': ['side table'], 'ArmChair': ['armchair', 'arm chair'],
    'Chair': ['chair'], 'CoffeeTable': ['coffee table'], 'Sofa': ['sofa', 'couch'],
    'Fridge': ['fridge', 'refrigerator'], 'Sink': ['sink'], 'Toilet': ['toilet'],
    'WashingMachine': ['washing machine'],
}


def _aabb_info(obj_meta: dict):
    aabb = obj_meta.get('raw_metadata', {}).get('axisAlignedBoundingBox', {})
    center = aabb.get('center') or obj_meta.get('position') or {}
    size, points = aabb.get('size', {}), aabb.get('cornerPoints', [])
    cx, cy, cz = (float(center.get(axis, 0)) for axis in ('x', 'y', 'z'))
    hx, hz = float(size.get('x', 0)) / 2, float(size.get('z', 0)) / 2
    return cx, cy, cz, hx, hz, max((p[1] for p in points), default=cy)


def _surface_matches_anchor(surface: str, anchor_type: str) -> bool:
    keywords = _ANCHOR_SURFACE_KEYWORDS.get(anchor_type)
    if keywords is None:
        keywords = [' '.join(re.findall('[A-Z][a-z]*', anchor_type)).lower()]
    return any(keyword in surface for keyword in keywords)


def _direction_vectors(entry: dict):
    cam_pos = entry.get('camera', {}).get('cam_pos', [0, 0, 0])
    landmark = entry.get('object_landmark', {}).get('position', {})
    cam = np.array([float(cam_pos[0]), 0.0, float(cam_pos[2])])
    target = np.array([float(landmark.get('x', 0)), 0.0, float(landmark.get('z', 0))])
    forward = target - cam
    norm = float(np.linalg.norm(forward))
    if norm > 1e-6:
        forward /= norm
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    norm = float(np.linalg.norm(right))
    if norm > 1e-6:
        right /= norm
    return {'right': right, 'left': -right, 'front': -forward}


def _bbox_boundary_xz(cx: float, cz: float, hx: float, hz: float,
                      direction: np.ndarray, offset: float = 0.2):
    dx, dz = float(direction[0]), float(direction[2])
    candidates = []
    if abs(dx) > 1e-6:
        candidates.append(hx / abs(dx))
    if abs(dz) > 1e-6:
        candidates.append(hz / abs(dz))
    distance = min(candidates) if candidates else 0.3
    return cx + (distance + offset) * dx, cz + (distance + offset) * dz


def _verified_gt_position(entry: dict, direction: str) -> dict | None:
    info = entry.get('stage3', {}).get(direction)
    if not info or info.get('verdict') != 'appropriate':
        return None
    surface = (info.get('surface') or '').lower()
    anchor_type = entry.get('landmark_type', '')
    cx, _cy, cz, hx, hz, top_y = _aabb_info(entry.get('object_landmark', {}))
    if direction == 'up':
        return {'x': cx, 'y': top_y + 0.2, 'z': cz}
    vector = _direction_vectors(entry).get(direction)
    if vector is None:
        return None
    if direction in ('left', 'right') and anchor_type in _GROUP_SPECIAL \
            and _surface_matches_anchor(surface, anchor_type):
        return {
            'x': cx + float(vector[0]) * 0.2,
            'y': top_y + 0.2,
            'z': cz + float(vector[2]) * 0.2,
        }
    surface_y = 0.0 if 'floor' in surface else top_y
    px, pz = _bbox_boundary_xz(cx, cz, hx, hz, vector)
    return {'x': px, 'y': surface_y + 0.2, 'z': pz}


def aggregate(evaluations: list[dict]) -> dict:
    successful = [item for item in evaluations if item.get('status') == 'ok']

    def stats(group: list[dict], total: int) -> dict:
        result = {
            'n': len(group), 'n_total': total,
            'mean_l2': None, 'std_l2': None, 'mean_abs_x': None,
            'mean_abs_y': None, 'pct_in_frame': None,
            'response_rate': len(group) / total if total else None,
        }
        if group:
            result.update({
                'mean_l2': float(np.mean([x['error_l2'] for x in group])),
                'std_l2': float(np.std([x['error_l2'] for x in group])),
                'mean_abs_x': float(np.mean(np.abs([x['error_x'] for x in group]))),
                'mean_abs_y': float(np.mean(np.abs([x['error_y'] for x in group]))),
                'pct_in_frame': float(np.mean([x['gt_in_frame'] for x in group])) * 100,
            })
        return result

    grouped_ok = {'type': {}, 'direction': {}}
    grouped_total = {'type': {}, 'direction': {}}
    for item in evaluations:
        for key in grouped_total:
            value = item.get(key)
            if value is not None:
                grouped_total[key][value] = grouped_total[key].get(value, 0) + 1
        if item.get('status') == 'ok':
            for key in grouped_ok:
                grouped_ok[key].setdefault(item[key], []).append(item)
    total = len(evaluations)
    return {
        'total': total,
        'n_ok': len(successful),
        'response_rate': len(successful) / total if total else None,
        'n_parse_error': sum(bool(item.get('parse_error')) for item in evaluations),
        'overall': stats(successful, total),
        'per_type': {
            key: stats(grouped_ok['type'].get(key, []), count)
            for key, count in sorted(grouped_total['type'].items())
        },
        'per_direction': {
            key: stats(grouped_ok['direction'].get(key, []), count)
            for key, count in sorted(grouped_total['direction'].items())
        },
    }


class POVBench(ImageBaseDataset):
    """The public POVBench train split (Inferred, Stated, and Given)."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    force_use_dataset_prompt = True
    DATASET_URL = {
        'POVBench': (
            'https://huggingface.co/datasets/owl-owl/POVBench/tree/'
            f'{POVBENCH_REVISION}/artifacts'
        )
    }

    def __init__(self, dataset='POVBench', root=None, sample_ids=None, skip_noimg=True):
        self._requested_root = root
        self._sample_ids = None if sample_ids is None else {str(x) for x in sample_ids}
        self._entries = {}
        self._exploration_logs = {}
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    @staticmethod
    def _artifacts_root(root: Path) -> Path:
        if any(path.is_dir() for path in root.glob('train_house_*')):
            return root
        if any(path.is_dir() for path in (root / 'artifacts').glob('train_house_*')):
            return root / 'artifacts'
        raise FileNotFoundError(
            f'POVBench root {root} has no train_house_* directories; expected the pinned artifacts snapshot.'
        )

    def _resolve_root(self) -> Path:
        configured = self._requested_root or os.environ.get('POVBENCH_ROOT')
        if configured:
            return self._artifacts_root(Path(configured).expanduser().resolve())

        local_root = Path(LMUDataRoot()) / 'POVBench'
        try:
            return self._artifacts_root(local_root)
        except FileNotFoundError:
            pass

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                'POVBench is not cached and automatic acquisition requires huggingface_hub.'
            ) from exc
        snapshot_download(
            repo_id=POVBENCH_REPO,
            repo_type='dataset',
            revision=POVBENCH_REVISION,
            allow_patterns=['artifacts/**'],
            local_dir=str(local_root),
        )
        return self._artifacts_root(local_root)

    def load_data(self, dataset):
        if dataset != 'POVBench':
            raise ValueError(f'Unsupported POVBench dataset name: {dataset}')
        self.artifacts_root = self._resolve_root()
        rows = []
        for house_dir in sorted(self.artifacts_root.glob('train_house_*')):
            if not house_dir.is_dir():
                continue
            source = house_dir / f'{house_dir.name}.json'
            if not source.is_file():
                raise FileNotFoundError(f'Missing POVBench house record: {source}')
            records = json.loads(source.read_text(encoding='utf-8'))
            if not isinstance(records, list):
                raise ValueError(f'POVBench house record must be a JSON list: {source}')
            exploration = sorted(
                path for path in (house_dir / 'exploration_images').glob('*.png')
                if 'depth' not in path.name
            )
            for entry in records:
                pair_id = str(entry.get('pair_id', ''))
                house_id = house_dir.name
                if not pair_id:
                    raise ValueError(f'POVBench entry without pair_id in {source}')
                entry = dict(entry)
                entry.setdefault('house_id', house_id)
                self._entries[(house_id, pair_id)] = entry
                directions = sorted(set(entry.get('stage5', {})) | set(entry.get('stage6', {})))
                for direction in directions:
                    s5 = entry.get('stage5', {}).get(direction, {})
                    s6 = entry.get('stage6', {}).get(direction, {})
                    for condition, sentence in (
                        ('a', s5.get('type_a', '')),
                        ('b', s5.get('type_b', '')),
                        ('c', s6.get('type_c', '')),
                    ):
                        if not sentence:
                            continue
                        sample_id = f'{house_id}|{pair_id}|{direction}|{condition}'
                        if self._sample_ids is not None and sample_id not in self._sample_ids:
                            continue
                        if condition == 'c':
                            rgb_path = entry.get('rgb_path')
                            images = [house_dir / rgb_path] if rgb_path else []
                        else:
                            images = exploration
                        missing = [str(path) for path in images if not path.is_file()]
                        if not images or missing:
                            detail = missing[0] if missing else f'{house_id}/{condition}'
                            raise FileNotFoundError(f'Missing POVBench media for {sample_id}: {detail}')
                        camera = entry.get('camera', {})
                        rows.append({
                            'index': sample_id,
                            'house_id': house_id,
                            'pair_id': pair_id,
                            'direction': direction,
                            'condition': condition,
                            'sentence': sentence,
                            'question': sentence,
                            'image_path': [str(path.resolve()) for path in images],
                            'candidate_images': [
                                f'exploration_images/{path.name}' for path in exploration
                            ] if condition != 'c' else [],
                            'screen_width': int(camera.get('screen_width', SCREEN_WIDTH)),
                            'screen_height': int(camera.get('screen_height', SCREEN_HEIGHT)),
                        })
        if not rows:
            raise ValueError(f'POVBench root produced no judgments: {self.artifacts_root}')
        frame = pd.DataFrame(rows)
        if frame['index'].duplicated().any():
            raise ValueError('POVBench stable judgment IDs are not unique')
        if self._sample_ids is not None:
            found = set(frame['index'])
            missing_ids = sorted(self._sample_ids - found)
            if missing_ids:
                raise KeyError(f'Unknown POVBench sample IDs: {missing_ids[:3]}')
        return frame

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        paths = line['image_path']
        if isinstance(paths, str):
            paths = [paths]
        condition = line['condition']
        messages = []
        if condition in ('a', 'b'):
            for index, path in enumerate(paths):
                messages.append({'type': 'text', 'value': f'Image {index}:'})
                messages.append({'type': 'image', 'value': path})
            text = prompt_ab(line['sentence'], len(paths))
        else:
            messages.extend((
                {'type': 'text', 'value': 'Image:'},
                {'type': 'image', 'value': paths[0]},
            ))
            text = prompt_c(line['sentence'], int(line['screen_width']), int(line['screen_height']))
        messages.append({'type': 'text', 'value': text})
        return messages

    def _exploration_log(self, house_id: str):
        if house_id not in self._exploration_logs:
            source = self.artifacts_root / house_id / 'exploration_log.json'
            records = json.loads(source.read_text(encoding='utf-8'))
            self._exploration_logs[house_id] = {
                item['image_path']: item for item in records if 'image_path' in item
            }
        return self._exploration_logs[house_id]

    def _evaluate_row(self, row: dict, raw_prediction: str) -> dict:
        condition = row['condition']
        parsed = parse_response(raw_prediction, condition)
        result = {
            'sample_id': row['index'], 'pair_id': row['pair_id'],
            'house_id': row['house_id'], 'direction': row['direction'],
            'type': condition, 'sentence': row['sentence'],
            'predicted_x': parsed['x'] if parsed else None,
            'predicted_y': parsed['y'] if parsed else None,
            'parse_error': None if parsed else 'failed to parse JSON',
        }
        if parsed is None:
            result['status'] = 'no_prediction'
            return result
        entry = self._entries.get((row['house_id'], row['pair_id']))
        if entry is None:
            result['status'] = 'entry_not_found'
            return result
        placement = entry.get('placements', {}).get(row['direction'])
        if not placement:
            result['status'] = 'placement_not_found'
            return result
        gt_position = placement.get('position')
        if not gt_position:
            result['status'] = 'no_gt_position'
            return result
        corrected = _verified_gt_position(entry, row['direction'])
        result['gt_surface_corrected'] = corrected is not None
        gt_position = corrected or gt_position

        if condition == 'c':
            camera = entry.get('camera', {})
            width = camera.get('screen_width', SCREEN_WIDTH)
            height = camera.get('screen_height', SCREEN_HEIGHT)
            result['camera_image'] = entry.get('rgb_path')
        else:
            index = parsed.get('selected_image')
            candidates = row['candidate_images']
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                result['status'] = 'no_selected_image'
                return result
            predicted_image = candidates[index]
            result['predicted_image'] = predicted_image
            log_entry = self._exploration_log(row['house_id']).get(predicted_image)
            if log_entry is None:
                result['status'] = 'image_not_in_log'
                return result
            camera = log_entry.get('camera', {})
            width, height = SCREEN_WIDTH, SCREEN_HEIGHT
            result['camera_image'] = predicted_image

        gx, gy, in_frame = project_3d_to_2d(
            gt_position,
            camera.get('cam_pos', [0, 0, 0]),
            camera.get('cam_yaw', 0.0),
            camera.get('cam_horizon', 0.0),
            fov_deg=camera.get('fov', 100.0),
            screen_width=width,
            screen_height=height,
        )
        px = float(np.clip(parsed['x'] / width, 0.0, 1.0))
        py = float(np.clip(parsed['y'] / height, 0.0, 1.0))
        result.update({
            'gt_position_3d': gt_position,
            'gt_position_2d': {'x': gx, 'y': gy},
            'gt_in_frame': in_frame,
            'predicted_x_norm': px,
            'predicted_y_norm': py,
            'error_x': px - gx,
            'error_y': py - gy,
            'error_l2': math.sqrt((px - gx) ** 2 + (py - gy) ** 2),
            'status': 'ok',
        })
        return result

    def evaluate_predictions(self, predictions: dict[str, str]):
        evaluations = []
        for row in self.data.to_dict('records'):
            raw = predictions.get(str(row['index']), '')
            evaluations.append(self._evaluate_row(row, str(raw)))
        return aggregate(evaluations), evaluations

    @staticmethod
    def metrics_from_summary(summary: dict) -> dict:
        metrics = {
            'response_rate/Overall': summary['response_rate'],
            'total': summary['total'],
            'n_ok': summary['n_ok'],
            'n_parse_error': summary['n_parse_error'],
        }
        for condition, display in CONDITION_NAMES.items():
            values = summary['per_type'].get(condition, {})
            metrics[f'mean_normalized_L2_error/{display}'] = values.get('mean_l2')
            metrics[f'response_rate/{display}'] = values.get('response_rate')
        return metrics

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame) or 'index' not in data or 'prediction' not in data:
            raise ValueError('POVBench evaluation requires index and prediction columns')
        predictions = {str(index): value for index, value in zip(data['index'], data['prediction'])}
        summary, evaluations = self.evaluate_predictions(predictions)
        detail_file = get_intermediate_file_path(eval_file, '_details', 'json')
        dump({'summary': summary, 'evaluations': evaluations}, detail_file)
        return self.metrics_from_summary(summary)

    @classmethod
    def report_primary_metric(cls, metrics):
        keys = [f'mean_normalized_L2_error/{name}' for name in CONDITION_NAMES.values()]
        return {key: metrics[key] for key in keys if metrics.get(key) is not None}


def load_prediction_directory(path: str | Path):
    """Load official prediction JSON with the documented merged-dir repair.

    If merged per-house lists exist, worker/base dictionaries are ignored. If
    no merged files exist, the official shard/base behavior is retained.
    """
    path = Path(path)
    if not path.is_dir():
        data = json.loads(path.read_text(encoding='utf-8'))
        return (data, {}) if isinstance(data, list) else (
            data.get('predictions', []), data.get('metadata', {})
        )
    files = [item for item in sorted(path.glob('*.json'))
             if item.name not in {'meta.json', 'eval_results.json'}]
    loaded = []
    for item in files:
        try:
            loaded.append((item, json.loads(item.read_text(encoding='utf-8'))))
        except Exception:
            continue
    merged = [(item, data) for item, data in loaded if isinstance(data, list)]
    selected = merged if merged else loaded
    predictions, metadata = [], {}
    for _item, data in selected:
        if isinstance(data, list):
            predictions.extend(data)
        elif isinstance(data, dict):
            predictions.extend(data.get('predictions', []))
            if not metadata:
                metadata = data.get('metadata') or {
                    key: data[key] for key in ('screen_width', 'screen_height', 'vlm', 'types')
                    if key in data
                }
    return predictions, metadata
