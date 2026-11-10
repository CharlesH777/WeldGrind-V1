import numpy as np

from weldgrind_v1.heightmap_planning import plan_field_program


def patch(hole=False):
    x, y = np.meshgrid(np.linspace(-2., 2., 41), np.linspace(-2., 2., 41))
    base = np.stack((x, y, np.zeros_like(x)), -1)
    normal = np.zeros_like(base); normal[..., 2] = 1.
    height = np.ones_like(x)
    if hole:
        height[x*x+y*y < .4**2] = -.3
    observed = base + height[..., None] * normal
    axis = np.zeros((2,) + x.shape); axis[0] = 1.
    return dict(cut_depth_mm=np.full(x.shape, .9), tangent_axis=axis, contact_probability=np.ones_like(x),
                points_world_mm=observed, base_anchors_world_mm=base, base_normals_world=normal,
                observation_valid=np.ones_like(x, bool), fit_supported=np.ones_like(x, bool),
                roi_mask=x*x+y*y < 1.5**2, process_params=np.array([.3, .4, 3., .5]))


def test_field_planner_layers_depth_and_keeps_commands_above_observed_base():
    source = patch(); copies = {k:v.copy() for k,v in source.items() if isinstance(v,np.ndarray)}
    result = plan_field_program(**source)
    phases = result['predicted_phases']; contacts = phases == 0
    assert result['report']['passes'] == 3
    assert result['report']['point_count'] == len(phases) > 20
    assert set(phases) == {0,1,2,3} and phases[0] == phases[-1] == 0
    assert result['trajectory_command_mm'][contacts,2].min() >= -1e-12
    assert result['cut_depth_mm'][contacts].max() <= .4
    assert not result['report']['uses_ground_truth'] and not result['report']['speed_magnitude_supervised']
    assert not result['report']['fixed_1024_claimed']
    for key,value in copies.items(): np.testing.assert_array_equal(source[key],value)


def test_every_phase_transition_is_connected_and_air_has_clearance():
    result = plan_field_program(**patch()); labels=result['predicted_phases']
    boundaries=np.flatnonzero(labels[:-1]!=labels[1:])
    for key in ['trajectory_reference_mm','trajectory_command_mm']:
        np.testing.assert_allclose(result[key][boundaries],result[key][boundaries+1],atol=1e-12)
    transfer=result['trajectory_command_mm'][labels==2]
    assert np.min(transfer[:,2]) >= 4.-1e-12
    assert np.all(result['cut_depth_mm'][labels!=0]==0)


def test_concavity_and_missing_support_never_request_cutting():
    source=patch(hole=True);result=plan_field_program(**source)
    contact=result['trajectory_command_mm'][result['predicted_phases']==0]
    assert np.min(np.linalg.norm(contact[:,:2],axis=1)) >= .39
    source['points_world_mm'][...,2]=-.3
    empty=plan_field_program(**source)
    assert len(empty['predicted_phases'])==0 and empty['end_token_index'] is None
    source=patch();source['fit_supported'][:]=False
    assert len(plan_field_program(**source)['predicted_phases'])==0


def test_planner_base_constraint_survives_rigid_coordinate_change():
    from scipy.spatial.transform import Rotation
    source=patch();rotation=Rotation.from_rotvec([.2,-.3,.4]).as_matrix();offset=np.array([20.,-10.,4.])
    for key in ['points_world_mm','base_anchors_world_mm']:
        source[key]=source[key]@rotation.T+offset
    source['base_normals_world']=source['base_normals_world']@rotation.T
    result=plan_field_program(**source)
    command=result['trajectory_command_mm'][result['predicted_phases']==0]
    local=(command-offset)@rotation
    assert local[:,2].min()>=-1e-10


def test_nearest_neighbour_enlarged_geometry_uses_nonzero_pixel_derivatives():
    source=patch()
    for key,value in list(source.items()):
        if key=='tangent_axis': source[key]=np.repeat(np.repeat(value,4,axis=1),4,axis=2)
        elif key!='process_params': source[key]=np.repeat(np.repeat(value,4,axis=0),4,axis=1)
    result=plan_field_program(**source)
    assert result['report']['status']=='planned' and len(result['predicted_phases'])>0
