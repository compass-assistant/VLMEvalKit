import json
import base64

import pandas as pd
import pytest

from vlmeval.dataset import POVBench, build_dataset
from vlmeval.dataset.povbench import aggregate, load_prediction_directory, parse_response
from vlmeval.api import GPT4V


def _write_fixture(root, *, with_observer_image=True):
    house = root / 'train_house_00000'
    images = house / 'exploration_images'
    images.mkdir(parents=True)
    (images / '000000.png').write_bytes(b'png')
    if with_observer_image:
        (house / 'observer.png').write_bytes(b'png')
    camera = {
        'cam_pos': [0.0, 1.0, -2.0], 'cam_yaw': 0.0,
        'cam_horizon': 0.0, 'fov': 100.0,
        'screen_width': 512, 'screen_height': 512,
    }
    entry = {
        'pair_id': 'pair-1', 'house_id': house.name, 'landmark_type': 'Desk',
        'camera': camera, 'rgb_path': 'observer.png',
        'stage3': {'right': {'verdict': 'appropriate', 'surface': 'floor'}},
        'stage5': {'right': {'type_a': 'sentence a', 'type_b': 'sentence b'}},
        'stage6': {'right': {'type_c': 'sentence c'}},
        'placements': {'right': {'position': {'x': 1.0, 'y': 0.2, 'z': 0.0}}},
        'object_landmark': {
            'position': {'x': 0.0, 'y': 0.5, 'z': 0.0},
            'raw_metadata': {'axisAlignedBoundingBox': {
                'center': {'x': 0.0, 'y': 0.5, 'z': 0.0},
                'size': {'x': 1.0, 'y': 1.0, 'z': 1.0},
                'cornerPoints': [[-0.5, 0.0, -0.5], [0.5, 1.0, 0.5]],
            }},
        },
    }
    (house / f'{house.name}.json').write_text(json.dumps([entry]))
    (house / 'exploration_log.json').write_text(json.dumps([{
        'image_path': 'exploration_images/000000.png', 'camera': camera,
    }]))
    (house / 'per_image_visible_objects.json').write_text('{}')


def test_registered_dataset_loads_real_row_shape_and_prompts(tmp_path):
    _write_fixture(tmp_path)
    dataset = build_dataset('POVBench', root=str(tmp_path))

    assert isinstance(dataset, POVBench)
    assert list(dataset.data['index']) == [
        'train_house_00000|pair-1|right|a',
        'train_house_00000|pair-1|right|b',
        'train_house_00000|pair-1|right|c',
    ]
    prompt = dataset.build_prompt(0)
    assert [item['type'] for item in prompt] == ['text', 'image', 'text']
    assert prompt[0]['value'] == 'Image 0:'
    assert '"selected_image"' in prompt[-1]['value']
    given = dataset.build_prompt(2)
    assert given[0] == {'type': 'text', 'value': 'Image:'}
    assert '"x"' in given[-1]['value'] and 'selected_image' not in given[-1]['value']


def test_parser_preserves_last_in_range_then_out_of_range_fallback():
    text = '{"x": 100, "y": 100} then {"x": 700, "y": 1}'
    assert parse_response(text, 'c') == {'x': 100.0, 'y': 100.0}
    assert parse_response('{"x": -2, "y": 800}', 'c') == {'x': -2.0, 'y': 800.0}
    assert parse_response('{not json}', 'c') is None


def test_evaluation_keeps_nonresponses_in_denominator(tmp_path):
    _write_fixture(tmp_path)
    dataset = POVBench(root=str(tmp_path))
    predictions = {
        dataset.data.iloc[0]['index']: '{"selected_image": 0, "x": 1, "y": 1}',
        dataset.data.iloc[1]['index']: 'malformed',
        dataset.data.iloc[2]['index']: '{"x": 2, "y": 2}',
    }
    summary, evaluations = dataset.evaluate_predictions(predictions)

    assert summary['total'] == 3
    assert summary['n_ok'] == 2
    assert summary['n_parse_error'] == 1
    assert summary['response_rate'] == pytest.approx(2 / 3)
    assert summary['per_type']['b']['response_rate'] == 0
    assert evaluations[1]['status'] == 'no_prediction'


def test_undefined_aggregation_states_are_not_manufactured():
    assert aggregate([])['response_rate'] is None
    only_failure = aggregate([{
        'type': 'a', 'direction': 'left', 'status': 'no_prediction',
        'parse_error': 'failed to parse JSON',
    }])
    assert only_failure['response_rate'] == 0
    assert only_failure['per_type']['a']['mean_l2'] is None


def test_merged_prediction_files_suppress_duplicate_shards(tmp_path):
    record = {'house_id': 'h', 'pair_id': 'p', 'direction': 'left', 'type': 'a'}
    (tmp_path / 'predictions_0.json').write_text(json.dumps({'predictions': [record]}))
    (tmp_path / 'train_house_00000.json').write_text(json.dumps([record]))

    predictions, _ = load_prediction_directory(tmp_path)
    assert predictions == [record]


def test_missing_media_and_unsupported_dataset_fail_clearly(tmp_path):
    _write_fixture(tmp_path, with_observer_image=False)
    with pytest.raises(FileNotFoundError, match='Missing POVBench media'):
        POVBench(root=str(tmp_path))
    with pytest.raises(ValueError, match='Unsupported POVBench dataset name'):
        POVBench(dataset='POVBench-invalid', root=str(tmp_path))


def test_native_evaluate_writes_details(tmp_path):
    _write_fixture(tmp_path)
    dataset = POVBench(root=str(tmp_path))
    prediction_file = tmp_path / 'predictions.pkl'
    pd.DataFrame({
        'index': list(dataset.data['index']),
        'prediction': [
            '{"selected_image": 0, "x": 10, "y": 10}',
            '{"selected_image": 0, "x": 20, "y": 20}',
            '{"x": 30, "y": 30}',
        ],
    }).to_pickle(prediction_file)

    metrics = dataset.evaluate(str(prediction_file))
    assert set(name for name in metrics if name.startswith('mean_normalized')) == {
        'mean_normalized_L2_error/Inferred',
        'mean_normalized_L2_error/Stated',
        'mean_normalized_L2_error/Given',
    }
    assert (tmp_path / 'predictions_details.json').is_file()


def test_openai_client_can_preserve_png_without_detail_hint(tmp_path):
    from PIL import Image

    image = tmp_path / 'input.png'
    Image.new('RGB', (128, 128), 'white').save(image)
    model = GPT4V(
        model='gpt-5.4', key='test', api_base='http://provider.invalid/chat/completions',
        image_format='PNG', img_detail=None, verbose=False,
    )
    content = model.prepare_itlist([{'type': 'image', 'value': str(image)}])
    image_url = content[0]['image_url']
    assert set(image_url) == {'url'}
    encoded = image_url['url'].removeprefix('data:image/png;base64,')
    assert base64.b64decode(encoded).startswith(b'\x89PNG\r\n\x1a\n')
