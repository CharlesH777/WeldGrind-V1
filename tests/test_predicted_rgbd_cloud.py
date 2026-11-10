"""No GPU, GT mask or old annotated point-cloud dependence in RGB-D input."""

import json

import numpy as np
import pytest
from PIL import Image

from weldgrind_v1.predicted_rgbd_cloud import (
    DEPTH_PREPROCESSING, POINTCLOUD_SOURCE, backproject_predicted_rgbd,
    fixed_cloud_seed, load_predicted_rgbd_cloud, mask_postprocess,
)


def camera(height,width,offset=.5):
    return {'capture_status':'accepted','world_units':'mm','image_size_wh':[width,height],
            'intrinsics_K':[[2,0,1],[0,4,.5],[0,0,1]],
            'world_from_camera_cv_mm':[[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]],
            'pixel_center_offset':offset,
            'depth_definition':'distance_to_image_plane, along optical +Z'}


def rgb_array(height,width):
    array=np.arange(height*width*3,dtype=np.uint8).reshape(height,width,3)
    return array


def test_half_pixel_backprojection_world_transform_and_color_alignment():
    rgb=rgb_array(2,3)
    depth=np.array([[8,12,16],[20,24,28]],dtype=np.float32)
    mask=np.array([[True,False,True],[False,True,False]])
    calib=camera(2,3)
    arrays,meta=backproject_predicted_rgbd(rgb,depth,calib,mask,context_max_points=99)
    uv=arrays['point_uv']
    u,v=uv.T
    expected_camera=np.column_stack([(u+.5-1)*depth[v,u]/2,(v+.5-.5)*depth[v,u]/4,depth[v,u]])
    expected_world=np.column_stack([-expected_camera[:,1]+10,expected_camera[:,0]+20,expected_camera[:,2]+30])
    np.testing.assert_allclose(arrays['xyz_camera_cv_mm'],expected_camera)
    np.testing.assert_allclose(arrays['points'][:,:3],expected_world)
    np.testing.assert_allclose(arrays['points'][:,3:],rgb[v,u]/255)
    np.testing.assert_array_equal(arrays['roi_mask'],mask[v,u])
    assert arrays['pointcloud_source'].item()==POINTCLOUD_SOURCE
    assert meta['pixel_center_offset']==.5 and meta['source_depth_is_object_foreground_cropped']
    assert meta['source_depth_preprocessing']==DEPTH_PREPROCESSING
    assert meta['unique_pixel_count']==meta['point_count']==6


def test_offset_is_used_exactly_once():
    rgb=rgb_array(1,1);depth=np.array([[8]],np.float32);mask=np.ones((1,1),bool)
    centered,_=backproject_predicted_rgbd(rgb,depth,camera(1,1,.5),mask)
    integer,_=backproject_predicted_rgbd(rgb,depth,camera(1,1,0),mask)
    difference=centered['xyz_camera_cv_mm']-integer['xyz_camera_cv_mm']
    np.testing.assert_allclose(difference,[[2,1,0]])


def test_only_finite_positive_noisy_depth_is_valid_and_invalid_mask_pixels_are_counted():
    depth=np.array([[1,0,np.nan],[np.inf,-2,3]],np.float32)
    mask=np.array([[True,True,True],[False,True,False]])
    arrays,meta=backproject_predicted_rgbd(rgb_array(2,3),depth,camera(2,3),mask)
    assert meta['valid_depth_pixel_count']==2
    assert meta['predicted_mask_pixel_count']==4
    assert meta['predicted_mask_invalid_depth_pixel_count']==3
    assert meta['valid_predicted_roi_pixel_count']==1
    assert meta['sampled_predicted_roi_point_count']==1 and meta['sampled_context_point_count']==1
    assert np.isfinite(arrays['points']).all()


def test_small_roi_is_not_duplicated_to_fill_a_quota():
    mask=np.zeros((10,10),bool);mask[4,5]=True
    arrays,meta=backproject_predicted_rgbd(rgb_array(10,10),np.ones((10,10)),camera(10,10),mask,
                                       roi_max_points=16384,context_max_points=8192)
    assert arrays['roi_mask'].sum()==1
    assert meta['valid_predicted_roi_pixel_count']==meta['sampled_predicted_roi_point_count']==1
    assert meta['unique_pixel_count']==100 and len(arrays['points'])==100


def test_capped_sampling_is_deterministic_unique_and_context_excludes_all_predicted_roi():
    height,width=30,40
    mask=np.zeros((height,width),bool);mask[:20]=True
    args=(rgb_array(height,width),np.ones((height,width)),camera(height,width),mask)
    a,meta=backproject_predicted_rgbd(*args,roi_max_points=100,context_max_points=50,seed=17)
    b,_=backproject_predicted_rgbd(*args,roi_max_points=100,context_max_points=50,seed=17)
    c,_=backproject_predicted_rgbd(*args,roi_max_points=100,context_max_points=50,seed=18)
    np.testing.assert_array_equal(a['point_uv'],b['point_uv'])
    assert not np.array_equal(a['point_uv'],c['point_uv'])
    assert len(np.unique(a['point_uv'],axis=0))==150
    assert a['roi_mask'].sum()==100
    assert not mask[a['point_uv'][100:,1],a['point_uv'][100:,0]].any()
    assert meta['valid_predicted_roi_pixel_count']==800
    # Independent context RNG is unchanged by a different ROI quota.
    d,_=backproject_predicted_rgbd(*args,roi_max_points=200,context_max_points=50,seed=17)
    np.testing.assert_array_equal(a['point_uv'][100:],d['point_uv'][200:])


def test_empty_or_all_invalid_roi_fails_without_any_legacy_fallback():
    mask=np.zeros((2,3),bool)
    with pytest.raises(ValueError,match='no finite positive-depth ROI'):
        backproject_predicted_rgbd(rgb_array(2,3),np.ones((2,3)),camera(2,3),mask)
    mask[0,0]=True;depth=np.ones((2,3));depth[0,0]=np.nan
    with pytest.raises(ValueError,match='no finite positive-depth ROI'):
        backproject_predicted_rgbd(rgb_array(2,3),depth,camera(2,3),mask)


@pytest.mark.parametrize('problem',['mask_dtype','depth_shape','offset','extrinsic','depth_definition'])
def test_malformed_calibration_and_inputs_are_rejected(problem):
    rgb=rgb_array(2,3);depth=np.ones((2,3));mask=np.ones((2,3),bool);calib=camera(2,3)
    if problem=='mask_dtype': mask=mask.astype(float)
    if problem=='depth_shape': depth=np.ones((3,2))
    if problem=='offset': calib.pop('pixel_center_offset')
    if problem=='extrinsic': calib['world_from_camera_cv_mm'][0][0]=2
    if problem=='depth_definition': calib['depth_definition']='range along ray'
    with pytest.raises(ValueError):
        backproject_predicted_rgbd(rgb,depth,calib,mask)


def test_file_loader_reads_only_rgb_noisy_depth_and_calibration(tmp_path,monkeypatch):
    rgb=rgb_array(2,3);mask=np.array([[1,0,0],[0,1,0]],bool)
    Image.fromarray(rgb).save(tmp_path/'rgb.png')
    np.savez(tmp_path/'depth.npz',noisy_depth_mm=np.ones((2,3),np.float32),
             depth_mm=np.full((2,3),999,dtype=np.float32),valid=np.zeros((2,3),bool),
             noisy_valid=np.zeros((2,3),bool))
    (tmp_path/'camera.json').write_text(json.dumps(camera(2,3)))
    (tmp_path/'legacy_cloud.npz').write_bytes(b'not even a valid NPZ; must not be read')
    row={'sample_id':'sample','image_size_wh':[3,2],'paths':{
        'rgb':'rgb.png','depth':'depth.npz','camera':'camera.json','camera_pointcloud':'legacy_cloud.npz',
        'grinding_mask_png':'nonexistent_gt_mask.png'}}
    original_load=np.load
    loaded=[]
    def tracked_load(path,*args,**kwargs):
        loaded.append(str(path))
        assert 'legacy_cloud' not in str(path)
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',tracked_load)
    arrays,meta=load_predicted_rgbd_cloud(tmp_path,row,mask)
    assert len(loaded)==1 and loaded[0].endswith('depth.npz')
    assert meta['valid_depth_pixel_count']==6 and meta['sampled_predicted_roi_point_count']==2
    assert not meta['source_camera_pointcloud_read'] and not meta['gt_defect_or_instance_labels_read']
    assert np.all(arrays['xyz_camera_cv_mm'][:,2]==1)


def test_loader_requires_noisy_depth_instead_of_clean_or_legacy_cloud(tmp_path):
    Image.fromarray(rgb_array(2,3)).save(tmp_path/'rgb.png')
    (tmp_path/'camera.json').write_text(json.dumps(camera(2,3)))
    np.savez(tmp_path/'depth.npz',depth_mm=np.ones((2,3),np.float32))
    row={'sample_id':'sample','image_size_wh':[3,2],
         'paths':{'rgb':'rgb.png','depth':'depth.npz','camera':'camera.json'}}
    with pytest.raises(ValueError,match='lacks noisy_depth_mm'):
        load_predicted_rgbd_cloud(tmp_path,row,np.ones((2,3),bool))
    row['paths'].pop('depth')
    with pytest.raises(ValueError,match='requires source path depth'):
        load_predicted_rgbd_cloud(tmp_path,row,np.ones((2,3),bool))


def test_sample_id_seed_is_stable_but_differs_by_image():
    assert fixed_cloud_seed('one',17)==fixed_cloud_seed('one',17)
    assert fixed_cloud_seed('one',17)!=fixed_cloud_seed('two',17)


def test_single_pixel_is_preserved_and_dilated_to_three_by_three_without_opening():
    raw=np.zeros((9,9),dtype=bool);raw[4,4]=True
    before=raw.tobytes()
    processed,meta=mask_postprocess(raw)
    expected=np.zeros_like(raw);expected[3:6,3:6]=True
    np.testing.assert_array_equal(processed,expected)
    assert raw.tobytes()==before and raw.sum()==1
    assert meta['opening_iterations']==0 and meta['dilation_iterations']==1
    assert not meta['opening_applied'] and not meta['erosion_applied']
    assert meta['raw_pixel_count']==1 and meta['processed_pixel_count']==9
    assert meta['removed_raw_pixel_count']==0


def test_three_by_three_block_expands_to_five_by_five_and_keeps_raw_bytes():
    raw=np.zeros((9,9),dtype=bool);raw[3:6,3:6]=True
    before=raw.copy()
    processed,meta=mask_postprocess(raw)
    expected=np.zeros_like(raw);expected[2:7,2:7]=True
    np.testing.assert_array_equal(processed,expected)
    np.testing.assert_array_equal(raw,before)
    assert not np.shares_memory(raw,processed)
    assert meta['raw_pixel_count']==9 and meta['processed_pixel_count']==25


def test_dilation_is_clipped_at_image_boundary_without_wraparound():
    raw=np.zeros((5,6),dtype=bool);raw[0,0]=True
    processed,meta=mask_postprocess(raw)
    expected=np.zeros_like(raw);expected[:2,:2]=True
    np.testing.assert_array_equal(processed,expected)
    assert processed.shape==raw.shape and not processed[-1].any() and not processed[:,-1].any()
    assert meta['outside_image_value'] is False


def test_pointcloud_roi_is_selected_from_processed_mask_and_not_the_raw_one():
    raw=np.zeros((9,9),dtype=bool);raw[3:6,3:6]=True
    processed,_=mask_postprocess(raw)
    arrays,meta=backproject_predicted_rgbd(rgb_array(9,9),np.ones((9,9)),camera(9,9),processed)
    roi_pixels=arrays['point_uv'][arrays['roi_mask']]
    expected_vu=np.argwhere(processed)
    np.testing.assert_array_equal(roi_pixels,expected_vu[:,::-1])
    assert arrays['roi_mask'].sum()==25 and raw.sum()==9
    assert meta['valid_predicted_roi_pixel_count']==25


def test_empty_mask_remains_empty_and_is_explicitly_reported():
    raw=np.zeros((4,4),dtype=bool)
    processed,meta=mask_postprocess(raw)
    assert not processed.any() and meta['processed_roi_empty']
    assert meta['raw_pixel_count']==meta['processed_pixel_count']==0


def test_opening_is_explicitly_disabled_and_non_boolean_masks_rejected():
    with pytest.raises(ValueError,match='Opening is disabled'):
        mask_postprocess(np.ones((3,3),bool),opening_iterations=1)
    with pytest.raises(ValueError,match='bool'):
        mask_postprocess(np.ones((3,3),np.uint8))
