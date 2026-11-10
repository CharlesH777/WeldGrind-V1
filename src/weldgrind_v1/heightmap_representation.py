"""Observed RGB-D height tiles, surface fitting, and cached tile loading."""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass,field
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .background_surface import BackgroundSurfaceConfig,ObservedBackgroundSurface
from .predicted_rgbd_cloud import mask_postprocess

SCHEMA='weldgrind-observed-heightmap-c15-v1'
FEATURE_NAMES=('rgb_r','rgb_g','rgb_b','signed_height_over_scale','sam_roi','fit_confidence','depth_valid',
    'crop_u','crop_v','base_normal_world_x','base_normal_world_y','base_normal_world_z',
    'observed_xyz_world_x_relative_center_over_scale','observed_xyz_world_y_relative_center_over_scale',
    'observed_xyz_world_z_relative_center_over_scale')
PROCESS_NAMES=('tool_radius_over_scale','layer_depth_over_scale','clearance_over_scale','overlap')


def _fit_config():
    # The numerical surface fitter is run entirely in ROI-normalized coordinates.
    # Its historical *_mm field names therefore denote normalized units here.
    return BackgroundSurfaceConfig(annulus_gap_mm=.05,annulus_width_mm=.6,max_annulus_width_mm=1.5,
        noise_floor_mm=.003,max_residual_sigma_mm=.1,model_residual_floor_mm=.015,
        quadratic_min_absolute_improvement_mm=.005,max_query_normal_distance_mm=6.)


@dataclass(frozen=True)
class HeightmapConfig:
    tile_size:int=128
    crop_expansion:float=3.
    min_crop_side:int=64
    axis_min_coherence:float=.2
    gt_min_view_cosine:float=.1
    background:BackgroundSurfaceConfig=field(default_factory=_fit_config)

    def __post_init__(self):
        if isinstance(self.background,dict):
            object.__setattr__(self,'background',BackgroundSurfaceConfig(**self.background))
        if not isinstance(self.background,BackgroundSurfaceConfig):
            raise ValueError('Heightmap background configuration must be explicit')
        if not isinstance(self.tile_size,int) or self.tile_size<8 or not isinstance(self.min_crop_side,int) or self.min_crop_side<2:
            raise ValueError('Heightmap sizes must be positive integers')
        if not np.isfinite(self.crop_expansion) or self.crop_expansion<1:
            raise ValueError('SAM crop expansion must be >=1')
        if not 0<=self.axis_min_coherence<=1 or not 0<=self.gt_min_view_cosine<1:
            raise ValueError('Invalid axis/view threshold')


def _config(value):
    return value if isinstance(value,HeightmapConfig) else HeightmapConfig(**(value or {}))


def _camera(camera):
    k=np.asarray(camera.get('intrinsics_K'),float);t=np.asarray(camera.get('world_from_camera_cv_mm'),float)
    offset=camera.get('pixel_center_offset')
    if (k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all()
            or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1])
            or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-6)
            or not isinstance(offset,(int,float)) or not 0<=offset<1):
        raise ValueError('Explicit rigid optical-camera calibration is required')
    if camera.get('depth_definition')!='distance_to_image_plane, along optical +Z' or camera.get('world_units')!='mm':
        raise ValueError('Heightmap depth must be calibrated optical-Z millimetres')
    return k,t,float(offset)


def _backproject(depth,camera,u,v):
    k,t,offset=_camera(camera)
    rays=np.column_stack((u+offset,v+offset,np.ones(len(u))))@np.linalg.inv(k).T
    camera_xyz=rays*(np.asarray(depth,float)/rays[:,2])[:,None]
    return camera_xyz@t[:3,:3].T+t[:3,3]


def project_world_points(world,camera):
    """Return image pixel-centre coordinates (not array-index coordinates), Z."""
    k,t,_=_camera(camera);xyz=(np.asarray(world,float)-t[:3,3])@t[:3,:3]
    projected=xyz@k.T;z=xyz[:,2]
    uv=np.full((len(xyz),2),np.nan)
    good=np.isfinite(xyz).all(1)&(z>0)
    uv[good]=projected[good,:2]/projected[good,2:]
    return uv,z


def _crop(mask,config):
    v,u=np.nonzero(mask)
    if not len(u):raise ValueError('Actual SAM prediction is empty')
    width=int(u.max()-u.min()+1);height=int(v.max()-v.min()+1)
    side=int(np.ceil(max(config.min_crop_side,config.crop_expansion*max(width,height))))
    cx=(u.min()+u.max()+1)/2;cy=(v.min()+v.max()+1)/2
    x0=int(np.floor(cx-side/2));y0=int(np.floor(cy-side/2))
    return np.array([x0,y0,x0+side,y0+side],np.int64)


def build_heightmap_observation(rgb,noisy_depth_mm,camera,sam_prediction,*,config=None):
    """Build C15 without accepting any GT scene, target, clean depth or path.

    Fit support comes from raw crop pixels before128 resizing. Invalid physical
    grids contain NaN; NN features use finite placeholders plus explicit masks.
    Positive signed height faces the camera (convex positive, recess negative).
    """
    cfg=_config(config);rgb=np.asarray(rgb);depth=np.asarray(noisy_depth_mm,float);rawmask=np.asarray(sam_prediction)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or depth.shape!=rgb.shape[:2] or rawmask.shape!=depth.shape or rawmask.dtype!=bool:
        raise ValueError('RGB uint8, noisy optical depth and bool SAM must align')
    if camera.get('image_size_wh')!=[rgb.shape[1],rgb.shape[0]]:
        raise ValueError('RGBD and camera image sizes differ')
    _,transform,_=_camera(camera);mask,mask_report=mask_postprocess(rawmask)
    finite=np.isfinite(depth)&(depth>0);rv,ru=np.nonzero(mask&finite)
    if len(ru)<6:raise ValueError('Fewer than six observed SAM ROI pixels')
    roi_world=_backproject(depth[rv,ru],camera,ru,rv).astype(np.float32)
    center=((roi_world.min(0)+roi_world.max(0))*.5).astype(np.float32)
    scale=float(np.linalg.norm(roi_world-center,axis=1).max())
    if not np.isfinite(scale) or scale<=1e-8:raise ValueError('Degenerate observed ROI metric scale')
    bounds=_crop(mask,cfg);x0,y0,x1,y1=map(int,bounds);size=cfg.tile_size
    u=np.floor(x0+(np.arange(size)+.5)*(x1-x0)/size).astype(int)
    v=np.floor(y0+(np.arange(size)+.5)*(y1-y0)/size).astype(int)
    uu,vv=np.meshgrid(u,v);inside=(uu>=0)&(uu<depth.shape[1])&(vv>=0)&(vv<depth.shape[0])
    safeu=np.clip(uu,0,depth.shape[1]-1);safev=np.clip(vv,0,depth.shape[0]-1)
    depth_valid=inside&finite[safev,safeu]
    roi=inside&mask[safev,safeu]
    tile_rgb=rgb[safev,safeu].astype(np.float32)/255;tile_rgb[~inside]=0
    points=np.full((size,size,3),np.nan,float)
    points[depth_valid]=_backproject(depth[safev[depth_valid],safeu[depth_valid]],camera,safeu[depth_valid],safev[depth_valid])
    # The fit uses only observed crop pixels and the camera-oriented frame.
    cy,cx=np.nonzero(finite[max(0,y0):min(depth.shape[0],y1),max(0,x0):min(depth.shape[1],x1)])
    cy=cy+max(0,y0);cx=cx+max(0,x0)
    fit_world=_backproject(depth[cy,cx],camera,cx,cy)
    fit_points=(fit_world-center)/scale
    fit=ObservedBackgroundSurface(fit_points,mask[cy,cx],camera_position_mm=(transform[:3,3]-center)/scale,config=cfg.background)
    query=fit.query((points[depth_valid]-center)/scale)
    fit_supported=np.zeros((size,size),bool);fit_conf=np.zeros((size,size),np.float32)
    fit_supported[depth_valid]=query['supported']&(query['confidence']>0)
    fit_conf[depth_valid]=query['confidence'].astype(np.float32)
    signed=np.zeros((size,size),np.float32);normal=np.zeros((size,size,3),np.float32)
    supported_query=query['supported']&(query['confidence']>0)
    signed_values=np.where(supported_query,query['signed_distance_mm'],0)
    normals_values=np.where(supported_query[:,None],query['normals'],0)
    signed[depth_valid]=signed_values;normal[depth_valid]=normals_values
    anchors=np.full_like(points,np.nan);anchors[depth_valid]=query['anchors_mm']*scale+center;anchors[~fit_supported]=np.nan
    normal_grid=normal.astype(float);normal_grid[~fit_supported]=np.nan
    xyz_normalized=np.zeros_like(normal,np.float32);xyz_normalized[depth_valid]=((points[depth_valid]-center)/scale).astype(np.float32)
    cu,cv=np.meshgrid(2*(np.arange(size,dtype=np.float32)+.5)/size-1,2*(np.arange(size,dtype=np.float32)+.5)/size-1)
    features=np.concatenate((tile_rgb,signed[...,None],roi[...,None],fit_conf[...,None],depth_valid[...,None],
        cu[...,None],cv[...,None],normal,xyz_normalized),axis=-1).transpose(2,0,1).astype(np.float32)
    if features.shape!=(15,size,size) or not np.isfinite(features).all():raise ValueError('Invalid finite C15 observation')
    report={**fit.report,'units':'observed_ROI_scale_normalized',
        'fit_sigma_normalized':fit.report.get('fit_sigma_mm'),
        'fit_sigma_physical_mm':float(fit.report['fit_sigma_mm'])*scale if 'fit_sigma_mm' in fit.report else None,
        'confidence_policy':'normalized heatmap fit; not the original physical-mm simulator confidence qualification',
        'fit_coordinate_units':'observed_ROI_max_radius_normalized; historical mm config names denote normalized units',
        'background_config_normalized':asdict(cfg.background),'ground_truth_used':False,'SAM_postprocess':mask_report,
        'crop_rule':'square actual dilated SAM bbox expansion only, out-of-image padding explicitly invalid',
        'resize_rule':'nearest observed source pixel, never depth interpolation across missing pixels',
        'physical_grid_invalid_fill':'NaN; features use masked finite placeholders, not zero-height surface evidence'}
    return {'features':features,'points_world_mm':points.astype(np.float32),'base_anchors_world_mm':anchors.astype(np.float32),
        'base_normals_world':normal_grid.astype(np.float32),'depth_valid':depth_valid,'fit_supported':fit_supported,
        'observation_valid':depth_valid&fit_supported,'pixel_uv':np.stack((uu,vv),axis=-1).astype(np.int64),
        'crop_bounds_xyxy':bounds,'center_mm':center,'scale_mm':np.array(scale,np.float32),'sam_roi':roi,
        'fit_report':report}


def normalized_process_conditions(params,scale_mm):
    values=np.asarray(params,float)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.any(values[:3]<=0) or not 0<=values[3]<1 or not np.isfinite(scale_mm) or scale_mm<=0:
        raise ValueError('Known radius/depth/clearance/overlap and positive scale required')
    return np.r_[values[:3]/scale_mm,values[3]].astype(np.float32)






def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_observation_surface(row_or_cached_npz,*,surface_config=None,heightmap_config=None):
    """Load dense original noisy RGBD and return a world-mm observed surface.

    Accept a prepared program row/NPZ with its JSON sidecar, or a heightmap NPZ
    beside its index.json. Only RGB/noisy_depth/camera/SAM files are opened.
    ``surface_config`` uses PHYSICAL mm and defaults to the original physical
    BackgroundSurfaceConfig; this does not reuse normalized heatmap confidence
    or lower any simulator acceptance threshold. Query arbitrary WORLD-mm XYZ
    through result['surface'].query(...), including final paths/mesh vertices.
    """
    cfg=_config(heightmap_config)
    if isinstance(row_or_cached_npz,dict):
        record=dict(row_or_cached_npz);candidate=Path(record['npz']) if 'npz' in record else None
        if 'source_fingerprints' not in record and candidate is not None and candidate.with_suffix('.json').is_file():
            sidecar=json.loads(candidate.with_suffix('.json').read_text())
            if record.get('sample_id',sidecar.get('sample_id'))!=sidecar.get('sample_id'):
                raise ValueError('Manifest row and observation sidecar identities differ')
            record={**sidecar,**record}
    else:
        candidate=Path(row_or_cached_npz).resolve()
        if candidate.with_suffix('.json').is_file():record=json.loads(candidate.with_suffix('.json').read_text())
        else:
            index=json.loads(candidate.with_name('index.json').read_text())
            record=next((row for row in index['samples'] if row['npz']==candidate.name),None)
            if record is None:raise ValueError('No observation provenance for this cached NPZ')
    fingerprints=record.get('source_fingerprints',{});root=Path(record['source_root']).resolve() if 'source_root' in record else None
    paths={};used={}
    for key in ('rgb','depth','camera'):
        entry=fingerprints.get(key)
        if not isinstance(entry,dict):raise ValueError(f'Missing observed {key} provenance')
        path=Path(entry['path'])
        if not path.is_absolute():
            if root is None:raise ValueError('Relative observed source requires source_root')
            path=(root/path).resolve()
            if not path.is_relative_to(root):raise ValueError('Observed source path escapes its collection')
        if _sha(path)!=entry['sha256']:raise ValueError(f'Observed {key} SHA256 mismatch')
        paths[key]=path;used[key]={'path':str(path),'sha256':entry['sha256']}
    prediction=record.get('prediction') or fingerprints.get('SAM_prediction')
    if not isinstance(prediction,dict):raise ValueError('Actual SAM prediction provenance required')
    sam_path=Path(prediction['path'])
    if not sam_path.is_absolute() or _sha(sam_path)!=prediction['sha256']:raise ValueError('SAM prediction SHA256 mismatch')
    with np.load(sam_path,allow_pickle=False) as z:
        raw_mask=z['mask'].copy()
        if 'inference_executed' not in z or not bool(z['inference_executed'].item()):raise ValueError('SAM inference receipt is required')
        if str(z['rgb_sha256'].item())!=used['rgb']['sha256']:raise ValueError('SAM prediction refers to a different RGB')
    used['SAM_prediction']={'path':str(sam_path),'sha256':prediction['sha256']}
    rgb=np.asarray(Image.open(paths['rgb']).convert('RGB'));camera=json.loads(paths['camera'].read_text())
    with np.load(paths['depth'],allow_pickle=False) as z:depth=z['noisy_depth_mm'].copy()
    if depth.shape!=rgb.shape[:2] or raw_mask.shape!=depth.shape:raise ValueError('Dense observed RGBD/SAM size mismatch')
    mask,postprocess=mask_postprocess(raw_mask);bounds=_crop(mask,cfg);x0,y0,x1,y1=map(int,bounds)
    window=np.isfinite(depth[max(0,y0):min(depth.shape[0],y1),max(0,x0):min(depth.shape[1],x1)])
    window&=depth[max(0,y0):min(depth.shape[0],y1),max(0,x0):min(depth.shape[1],x1)]>0
    v,u=np.nonzero(window);u+=max(0,x0);v+=max(0,y0)
    world=_backproject(depth[v,u],camera,u,v);roi=mask[v,u].copy();physical_cfg=surface_config
    if physical_cfg is None:physical_cfg=BackgroundSurfaceConfig()
    surface=ObservedBackgroundSurface(world,roi,camera_position_mm=_camera(camera)[1][:3,3],config=physical_cfg)
    report={**surface.report,'dense_observation_source':'all finite positive original noisy-depth crop pixels before any resize',
        'no_clean_depth_or_GT_scene_or_path_read':True,'physical_surface_config':asdict(surface.config),
        'normalized_heightmap_confidence_reused':False,'SAM_postprocess':postprocess,'crop_bounds_xyxy':bounds.tolist(),
        'source_fingerprints':used}
    return {'surface':surface,'points_world_mm':world,'points_world_rgb':np.column_stack((world,rgb[v,u]/255.)),
        'roi_mask':roi,'pixel_uv':np.column_stack((u,v)).astype(np.int64),'camera':camera,
        'crop_bounds_xyxy':bounds,'source_fingerprints':used,'report':report}








class HeightmapDataset(Dataset):
    """Cache-only split-explicit dataset; source planner is never imported here."""
    def __init__(self,index_path,*,split):
        if split not in ('train','val','test'):raise ValueError('An explicit train/val/test split is required')
        self.index_path=Path(index_path).resolve();index=json.loads(self.index_path.read_text())
        if index.get('schema')!=SCHEMA or index.get('feature_names')!=list(FEATURE_NAMES):raise ValueError('Heightmap feature schema changed')
        rows=index.get('samples',[]);ids=[r['sample_id'] for r in rows]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample identities across cache splits')
        self.rows=[r for r in rows if r['split']==split]
        if not self.rows:raise ValueError('Requested split has no cached samples')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];relative=Path(row['npz'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Cache path must remain relative')
        path=(self.index_path.parent/relative).resolve()
        if not path.is_relative_to(self.index_path.parent) or _sha(path)!=row['sha256']:raise ValueError('Heightmap cache SHA256 mismatch')
        with np.load(path,allow_pickle=False) as z:sample={name:torch.from_numpy(z[name].copy()) for name in z.files}
        sample.update(sample_id=row['sample_id'],split=row['split'])
        if sample['features'].shape[0]!=15 or not torch.isfinite(sample['features']).all():raise ValueError('Invalid observation feature tensor')
        return sample




