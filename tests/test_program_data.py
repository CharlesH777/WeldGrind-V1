"""CPU tests of full-program labels; no GPU/SAM inference is performed."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from weldgrind_v1.collection import sha256_file
from weldgrind_v1.program_data import (
    PROGRAM_SCHEMA, ProgramDataset, _segment_modes, export_program_collection,
    freeze_program_selection, load_collection_program, resample_complete_program,
)


def program():
    curves = [np.array([[0,0,1],[.5,0,1.3],[1,0,1.0]]),
              np.array([[1,0,1],[1,0,6]]),
              np.array([[1,0,6],[1,1,6]]),
              np.array([[1,1,6],[1,1,.7]]),
              np.array([[1,1,.7],[0,1,.7]])]
    segments = []
    cursor = 0
    for i,curve in enumerate(curves):
        contact = i in (0,4)
        entry = {"start":cursor,"end":cursor+len(curve),"contact":contact,
                 "kind":"grind_stroke" if contact else "air_move"}
        if contact:
            entry.update(pass_index=i//4,surface_snapshot_index=i//4,
                         cut_level_chart_h_mm=.5 if i==0 else 0.0)
        segments.append(entry)
        cursor += len(curve)
    return np.concatenate(curves).astype(np.float32),segments


def test_full_sequence_keeps_all_air_modes_and_endpoints():
    raw,segments = program()
    result = resample_complete_program(raw,segments,count=64)
    assert result['trajectory'].shape == (64,3)
    assert set(result['trajectory_mode']) == {0,1,2,3}
    assert result['trajectory_contact'].sum() < 64
    assert result['trajectory_valid'].all()
    assert result['trajectory_stop'].sum() == 1 and result['trajectory_stop'][-1]
    for index,((begin,end),source) in enumerate(zip(result['resampled_segment_ranges'],segments)):
        np.testing.assert_array_equal(result['trajectory'][begin],raw[source['start']])
        np.testing.assert_array_equal(result['trajectory'][end-1],raw[source['end']-1])
        assert (result['trajectory_segment'][begin:end] == index).all()
    assert (result['trajectory_pass_index'][~result['trajectory_contact']] == -1).all()
    assert not result['trajectory_cut_level_valid'][~result['trajectory_contact']].any()


def test_cutting_plane_is_preserved_independently_from_contact_xyz():
    raw,segments=program()
    result=resample_complete_program(raw,segments,64)
    contact=result['trajectory_contact']
    assert np.max(np.abs(result['trajectory'][contact,2]-result['trajectory_cut_level_mm'][contact])) > .4
    assert set(result['trajectory_cut_level_mm'][contact]) == {0.0,.5}


def test_curvature_anchors_preserve_segment_internal_height_corner():
    x=np.linspace(0,100,1001)
    z=np.where(x<=33, x*.01, np.where(x<=33.1,.33-(x-33)*3.3,0))
    raw=np.column_stack([x,np.zeros_like(x),z])
    segments=[{'start':0,'end':len(raw),'contact':True,'kind':'grind_stroke',
               'pass_index':0,'surface_snapshot_index':0,'cut_level_chart_h_mm':0.0}]
    result=resample_complete_program(raw,segments,128,max_error_mm=.02)
    assert result['minimum_anchor_count'] > 2
    assert result['segment_reconstruction_max_error_mm'].max() <= .02
    assert np.any(np.isclose(result['trajectory'][:,0],33))
    assert np.any(np.isclose(result['trajectory'][:,0],33.1))


def test_insufficient_full_sequence_capacity_rejects_instead_of_dropping_motion():
    raw,segments=program()
    with pytest.raises(ValueError,match='capacity'):
        resample_complete_program(raw,segments,8)


@pytest.mark.parametrize('problem',['gap','contact_kind','missing_cut_level','disconnected','trailing_air'])
def test_malformed_program_semantics_are_rejected(problem):
    raw,segments=program()
    if problem=='gap': segments[1]['start']+=1
    elif problem=='contact_kind': segments[1]['contact']=True
    elif problem=='missing_cut_level': segments[0].pop('cut_level_chart_h_mm')
    elif problem=='disconnected': raw[segments[1]['start'],0]+=1
    else:
        segments=segments[:-1]
        raw=raw[:segments[-1]['end']]
    with pytest.raises(ValueError):
        resample_complete_program(raw,segments,64)


def test_air_auxiliary_modes_are_verified_in_chart_normal_direction():
    raw,segments=program()
    assert _segment_modes(segments,raw,np.array([0,0,1])).tolist()==[0,1,2,3,0]
    wrong=raw.copy()
    wrong[segments[1]['end']-1,0]+=1
    with pytest.raises(ValueError,match='retract'):
        _segment_modes(segments,wrong,np.array([0,0,1]))


@pytest.fixture
def collection(tmp_path):
    from PIL import Image
    root=tmp_path/'source'
    root.mkdir()
    rows=[]
    raw,segments=program()
    for i,split in enumerate(['train','test']):
        directory=root/split/f'sample_{i}'
        directory.mkdir(parents=True)
        Image.fromarray(np.full((2,3,3),128,dtype=np.uint8)).save(directory/'rgb.png')
        camera={'capture_status':'accepted','world_units':'mm','image_size_wh':[3,2],
                'world_from_camera_cv_mm':np.eye(4).tolist(),'intrinsics_K':np.eye(3).tolist(),
                'pixel_center_offset':.5,'depth_definition':'distance_to_image_plane, along optical +Z'}
        (directory/'camera.json').write_text(json.dumps(camera))
        np.savez(directory/'depth.npz',noisy_depth_mm=np.ones((2,3),np.float32))
        np.savez(directory/'cloud.npz',xyz_world_mm=np.array([[0,0,1],[1,0,1],[0,1,1]],np.float32),
                 xyz_camera_cv_mm=np.array([[0,0,1],[1,0,1],[0,1,1]],np.float32),
                 rgb=np.full((3,3),128,np.uint8),pixel_uv=np.array([[0,0],[1,0],[0,1]],np.int32))
        np.save(directory/'trajectory.npy',raw)
        metadata={'status':'validated','units':'mm','coordinate_frame':'world',
                  'representation':'ordered_xyz_surface_contact_reference',
                  'segment_ranges_convention':'half-open [start, end)',
                  'point_count':len(raw),'segment_ranges':segments,
                  'contact_reference_definition':'surface projection before each pass; not TCP',
                  'tool':{'clearance_mm':6}}
        (directory/'trajectory_meta.json').write_text(json.dumps(metadata))
        paths={name:str((directory/file).relative_to(root)) for name,file in {
            'rgb':'rgb.png','depth':'depth.npz','camera':'camera.json','camera_pointcloud':'cloud.npz',
            'trajectory_world':'trajectory.npy','trajectory_meta':'trajectory_meta.json'}.items()}
        rows.append({'sample_id':f'sample_{i}','split':split,'grindable':True,'units':'mm',
                     'coordinate_frame':'world','shape':'butt_plate','defect_code':502,
                     'image_size_wh':[3,2],'paths':paths})
    index=root/'training_index.jsonl'
    index.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (root/'training_bundle_manifest.json').write_text(json.dumps({'status':'complete','index':index.name,
            'index_sha256':sha256_file(index),'sample_count':len(rows)}))
    (root/'training_bundle_validation.json').write_text(json.dumps({'passed':True}))
    return root,rows,tmp_path


def predictions(root,rows,output):
    output.mkdir()
    for row in rows:
        np.savez(output/f"{row['sample_id']}.npz",mask=np.ones((2,3),dtype=bool),
                 sample_id=np.array(row['sample_id']),source=np.array('sam3_prediction'),
                 inference_executed=np.array(True),rgb_sha256=np.array(sha256_file(root/row['paths']['rgb'])),
                 checkpoint_sha256=np.array('a'*64))
    return output


def test_all_source_splits_are_explicitly_frozen_for_memorization(collection):
    root,rows,tmp=collection
    selection=freeze_program_selection(root,tmp/'selection.json')
    value=json.loads(selection.read_text())
    assert {row['split'] for row in value['selected']}=={'train','test'}
    assert value['include_all_source_splits'] and not value['generalization_claim']
    assert freeze_program_selection(root,selection)==selection
    with pytest.raises(ValueError,match='read-only'):
        freeze_program_selection(root,root/'forbidden.json')


def test_export_and_dataset_preserve_full_program_and_target_only_labels(collection):
    root,rows,tmp=collection
    selection=freeze_program_selection(root,tmp/'selection.json')
    pred=predictions(root,rows,tmp/'predictions')
    manifest=export_program_collection(root,selection,pred,tmp/'export',trajectory_points=64)
    dataset=ProgramDataset(manifest,max_points=8,trajectory_points=64)
    assert len(dataset)==2
    sample=dataset[1]
    assert sample['source_split']=='test'
    assert sample['trajectory'].shape==(64,3)
    assert set(sample['phase_labels'].tolist())=={0,1,2,3}
    assert torch.equal(sample['phase_labels'],sample['trajectory_mode'])
    assert torch.equal(sample['segment_ids'],sample['trajectory_segment'])
    restored=(sample['trajectory']*sample['scale']+sample['center']).numpy()
    raw,segments=program()
    np.testing.assert_allclose(restored[[0,-1]],raw[[0,-1]],atol=1e-6)
    with np.load(manifest.parent/'sample_1.npz',allow_pickle=False) as stored:
        assert stored['program_schema'].item()==PROGRAM_SCHEMA
        np.testing.assert_array_equal(stored['raw_trajectory_mm'],raw)
        assert stored['segment_trajectories_mm'].shape==(5,128,3)
        assert len(json.loads(stored['segment_ranges'].item()))==5
    receipt=json.loads((manifest.parent/'export_receipt.json').read_text())
    assert receipt['count']==2 and receipt['all_source_splits_used_for_overfit']
    assert not receipt['gt_mask_used_as_input'] and not receipt['source_program_has_final_retract']
    assert not receipt['source_camera_pointcloud_read'] and not receipt['gt_defect_or_instance_sampling_used']
    assert receipt['source_depth_is_object_foreground_cropped']
    assert receipt['pointcloud_statistics_per_sample'][0]['valid_predicted_roi_pixel_count']==6
    assert receipt['roi_mask_postprocess']['opening_iterations']==0
    assert receipt['roi_mask_postprocess']['dilation_iterations']==1
    assert receipt['sam_raw_prediction_and_iou_evaluation_unchanged']


def test_export_preserves_raw_sam_files_and_uses_only_dilated_downstream_roi(collection):
    root,rows,tmp=collection
    selection=freeze_program_selection(root,tmp/'selection.json')
    pred=predictions(root,rows,tmp/'predictions')
    raw=np.zeros((2,3),dtype=bool);raw[0,1]=True
    before={}
    for row in rows:
        path=pred/f"{row['sample_id']}.npz"
        with np.load(path,allow_pickle=False) as saved:
            arrays={key:saved[key] for key in saved.files}
        arrays['mask']=raw
        np.savez(path,**arrays)
        before[row['sample_id']]=path.read_bytes()
    manifest=export_program_collection(root,selection,pred,tmp/'export',trajectory_points=64)
    for row in rows:
        assert (pred/f"{row['sample_id']}.npz").read_bytes()==before[row['sample_id']]
        with np.load(manifest.parent/f"{row['sample_id']}.npz",allow_pickle=False) as exported:
            np.testing.assert_array_equal(exported['raw_sam_mask'],raw)
            assert exported['mask'].all() and exported['roi_mask'].sum()==6
            meta=json.loads(exported['mask_postprocess_metadata'].item())
            assert meta['raw_pixel_count']==1 and meta['processed_pixel_count']==6
            assert meta['operations']==['binary_dilation']
            assert meta['opening_iterations']==0 and meta['dilation_iterations']==1
            assert not meta['opening_applied'] and not meta['erosion_applied']


def test_export_never_reads_or_fingerprints_legacy_gt_sampled_cloud(collection,monkeypatch):
    root,rows,tmp=collection
    for row in rows:
        (root/row['paths']['camera_pointcloud']).write_bytes(b'poisoned legacy GT cloud must never be opened')
    original_load=np.load
    def checked_load(path,*args,**kwargs):
        if str(path).endswith('cloud.npz'):
            raise AssertionError('Legacy GT-defect-sampled cloud was accessed')
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',checked_load)
    selection=freeze_program_selection(root,tmp/'new_program_selection.json')
    frozen=json.loads(selection.read_text())
    assert all('depth' in row['source_fingerprints'] and 'camera_pointcloud' not in row['source_fingerprints']
               for row in frozen['selected'])
    pred=predictions(root,rows,tmp/'predictions')
    manifest=export_program_collection(root,selection,pred,tmp/'export',trajectory_points=64)
    assert len(ProgramDataset(manifest,trajectory_points=64))==2


def test_missing_depth_is_not_replaced_by_legacy_camera_pointcloud(collection):
    root,rows,tmp=collection
    index=root/'training_index.jsonl'
    for row in rows:
        row['paths'].pop('depth')
    index.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    bundle=json.loads((root/'training_bundle_manifest.json').read_text())
    bundle['index_sha256']=sha256_file(index)
    (root/'training_bundle_manifest.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='requires depth'):
        freeze_program_selection(root,tmp/'program_selection.json')


def test_export_requires_every_actual_sam_prediction_and_never_uses_gt_fallback(collection):
    root,rows,tmp=collection
    selection=freeze_program_selection(root,tmp/'selection.json')
    pred=tmp/'empty_predictions';pred.mkdir()
    with pytest.raises(FileNotFoundError,match='Actual SAM'):
        export_program_collection(root,selection,pred,tmp/'export',trajectory_points=64)
    assert not (tmp/'export').exists()


def test_program_dataset_rejects_gt_roi_or_unversioned_program(collection):
    root,rows,tmp=collection
    selection=freeze_program_selection(root,tmp/'selection.json')
    pred=predictions(root,rows,tmp/'predictions')
    manifest=export_program_collection(root,selection,pred,tmp/'export',trajectory_points=64)
    path=manifest.parent/'sample_0.npz'
    with np.load(path,allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    arrays['roi_source']=np.array('ground_truth')
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='actual SAM ROI'):
        ProgramDataset(manifest,trajectory_points=64)[0]
