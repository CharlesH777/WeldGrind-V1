"""User-facing pictures of actual raw full-program inference, including air moves."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def render_program_previews(summary: dict, manifest, output_dir, *, step: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image,ImageDraw

    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest=Path(manifest).resolve()
    inputs={r['sample_id']:r for r in (json.loads(x) for x in manifest.read_text().splitlines() if x.strip())}
    predictions={r['sample_id']:r for r in summary['per_image']}
    preferred=['workpiece_001_state_01','workpiece_002_state_01','workpiece_003_state_01',
               'workpiece_001_state_08','workpiece_011_state_10','workpiece_021_state_10']
    chosen=[sid for sid in preferred if sid in predictions]
    if not chosen: chosen=list(predictions)[:6]
    colors=np.array(['#168b56','#d54d42','#306bb2','#b37c21'])
    labels=['贴面打磨','退刀','高位横移','进刀']
    records=[]
    for sid in chosen:
        result=predictions[sid]
        with np.load(result['prediction'],allow_pickle=False) as data:
            pred=data['trajectory_raw_mm'];target=data['target_mm']
            phase=data['predicted_phases'];truth=data['target_phases'];segments=data['segment_ids']
        source=Path(inputs[sid]['npz'])
        if not source.is_absolute():source=manifest.parent/source
        with np.load(source,allow_pickle=False) as data:
            cloud=data['points'][:,:3]
        margin=np.maximum(np.ptp(np.concatenate([pred,target]),axis=0)*.15,1.)
        low=np.minimum(pred.min(0),target.min(0))-margin
        high=np.maximum(pred.max(0),target.max(0))+margin
        nearby=cloud[((cloud>=low)&(cloud<=high)).all(1)]
        if len(nearby)>4000:nearby=nearby[np.linspace(0,len(nearby)-1,4000,dtype=int)]
        fig=plt.figure(figsize=(12.6,5.6),dpi=150)
        ax=fig.add_subplot(1,2,1,projection='3d')
        if len(nearby):ax.scatter(*nearby.T,s=.3,color='#9ba5b0',alpha=.2,rasterized=True)
        ax.plot(*target.T,'--',color='#6d7480',lw=.8,alpha=.8)
        ax.add_collection3d(Line3DCollection(np.stack([pred[:-1],pred[1:]],1),colors=colors[phase[:-1]],linewidths=1.4))
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),
               xlabel='X / mm',ylabel='Y / mm',zlabel='Z / mm',title='原始完整程序预测（含空移，不贴面投影）')
        ax.set_box_aspect(np.maximum(high-low,.1))
        ax.view_init(elev=24,azim=-58)
        height_ax=fig.add_subplot(1,2,2)
        retract=np.flatnonzero(truth==1)
        if len(retract):
            segment=segments[retract[0]]
            indices=np.flatnonzero(segments==segment)
            normal=target[indices[-1]]-target[indices[0]]
            normal=normal/max(np.linalg.norm(normal),1e-12)
        else:normal=np.array([0.,0.,1.])
        origin=target[0]@normal
        y=pred@normal-origin;gt=target@normal-origin;x=np.arange(len(pred))
        height_ax.plot(x,gt,'--',color='#6d7480',lw=1,label='标注高度')
        height_ax.add_collection(LineCollection(np.stack([np.column_stack([x[:-1],y[:-1]]),
                  np.column_stack([x[1:],y[1:]])],1),colors=colors[phase[:-1]],linewidths=1.3))
        height_ax.autoscale();height_ax.grid(alpha=.18)
        height_ax.set(xlabel='序列位置（不是加工时间）',ylabel='相对加工法向高度 / mm',title='多遍接触与退刀高度')
        metrics=result['metrics']
        fig.suptitle(f'{sid}  |  step {step}  |  ordered RMSE {metrics["ordered_rmse_mm"]:.3f} mm  |  phase {metrics["phase_accuracy"]:.1%}',fontsize=11)
        legend=[Line2D([0],[0],color=c,lw=2,label=l) for c,l in zip(colors,labels)]
        legend.append(Line2D([0],[0],color='#6d7480',ls='--',label='标注参考'))
        fig.legend(handles=legend,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.005))
        fig.subplots_adjust(left=.02,right=.985,top=.86,bottom=.17,wspace=.18)
        path=output/f'{sid}_program_prediction.png'
        fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
        records.append({'sample_id':sid,'step':step,'image':str(path),'prediction':result['prediction'],
                        'ordered_rmse_mm':metrics['ordered_rmse_mm'],'phase_accuracy':metrics['phase_accuracy'],
                        'actual_raw_inference':True,'snap_to_surface':False})
    if records:
        thumbnails=[]
        for row in records:
            image=Image.open(row['image']).convert('RGB');image.thumbnail((1100,490));thumbnails.append(image)
        overview=Image.new('RGB',(1100,490*len(thumbnails)),'white')
        for i,image in enumerate(thumbnails):overview.paste(image,(0,490*i))
        overview.save(output/'program_predictions_overview.jpg',quality=94)
    (output/'preview_index.json').write_text(json.dumps({'step':step,'predictions':records,
         'actual_prediction':True,'color_is_predicted_phase':True,'scope':'complete labelled program; no initial approach/final retract labels'},indent=2,ensure_ascii=False)+'\n')
    return records
