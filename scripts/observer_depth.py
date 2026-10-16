"""Isolated D435 engineering capture/export. Never imported by production.

No robot serial, local control API, Astra or Jev is used. A live camera opens
only with explicit --live --serial; --bag reads an explicitly supplied SDK
recording. RGB/depth remain in observer optical coordinates until calibrated.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def select_pair(pipeline,seconds,*,playback=None,stopped_status=None):
    """Bounded live warmup or recorded-time selection, retaining valid EOF data."""
    deadline=time.monotonic()+seconds;frames=None;first_sensor_ms=None
    for _ in range(75):
        try:candidate=pipeline.wait_for_frames(1000)
        except RuntimeError:
            if playback and frames and playback.current_status()==stopped_status:break
            raise
        if not candidate.get_depth_frame() or not candidate.get_color_frame():continue
        frames=candidate;sensor_ms=frames.get_depth_frame().get_timestamp()
        if first_sensor_ms is None:first_sensor_ms=sensor_ms
        if (playback and sensor_ms-first_sensor_ms>=seconds*1000) or (not playback and time.monotonic()>=deadline):break
    if frames is None:raise RuntimeError('No paired depth and color frames were received')
    return frames


def export_observation(destination, *, depth_raw, color_rgb, points_m, metadata, aligned_depth_raw=None):
    """Save native measurements and a bounded parameterized text summary.

    points_m is the SDK-deprojected native depth cloud (distortion handled by
    SDK), not a pinhole approximation applied to an uncorrected RGB image.
    Summary extents describe visible samples, NOT object dimensions.
    """
    required={'depth_scale_m','depth_intrinsics','color_intrinsics','depth_to_color_extrinsics',
              'device_serial','device_name','depth_timestamp','color_timestamp','host_receipt_monotonic'}
    if not isinstance(metadata,dict) or not required.issubset(metadata):raise ValueError('Missing depth provenance')
    scale=metadata['depth_scale_m']
    if type(scale) not in (int,float) or not math.isfinite(scale) or not 0<scale<.1:raise ValueError('Invalid depth scale')
    if depth_raw.dtype!=np.uint16 or depth_raw.ndim!=2 or depth_raw.size>1920*1080:raise ValueError('Expected bounded native uint16 depth')
    h,w=depth_raw.shape
    if color_rgb.dtype!=np.uint8 or color_rgb.ndim!=3 or color_rgb.shape[2]!=3 or color_rgb.size>1920*1080*3:raise ValueError('Expected bounded RGB image')
    if points_m.shape!=(h*w,3):raise ValueError('Native points must correspond to native depth pixels')
    intrinsics=metadata['depth_intrinsics']
    if intrinsics.get('width')!=w or intrinsics.get('height')!=h:raise ValueError('Depth intrinsics do not match native image')
    valid=(depth_raw.reshape(-1)>0)&np.isfinite(points_m).all(axis=1)&(points_m[:,2]>0)
    if valid.any() and not np.allclose(points_m[valid,2],depth_raw.reshape(-1)[valid]*scale,rtol=1e-3,atol=1e-5):
        raise ValueError('Point cloud depth does not match native depth scale')
    if aligned_depth_raw is not None and (aligned_depth_raw.dtype!=np.uint16 or aligned_depth_raw.shape!=color_rgb.shape[:2]):
        raise ValueError('Aligned depth must match color dimensions')
    selected=np.flatnonzero(valid)
    if len(selected)>2048:selected=selected[np.linspace(0,len(selected)-1,2048,dtype=int)]
    points=points_m[selected]
    geometric=None if not len(points) else {'sample_count':len(points),
        'visible_surface_bounds_m':{'minimum':points.min(axis=0).round(5).tolist(),'maximum':points.max(axis=0).round(5).tolist()},
        'median_visible_depth_m':float(np.median(points[:,2])),
        'reference':'visible_samples_not_object_center_or_dimensions'}
    summary={'version':1,'source':'engineering_observer','production_eligible':False,
        'coordinate_frame':'observer_depth_optical','units':'m',
        'robot_base_transform':None,'registration_status':'not_calibrated',
        'camera_metadata':metadata,'valid_depth_fraction':float(np.mean(valid)),
        'geometry':geometric,'object_identities':[],'object_dimensions_m':None,
        'not_for_motion_authorization':True}
    # Serialize first so invalid metadata cannot leave a misleading partial report.
    encoded=json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    arrays={'depth_raw':depth_raw,'color_rgb':color_rgb,'native_valid_mask':valid.reshape(h,w),
            'sample_pixel_indices':selected,'sample_points_m':points}
    if aligned_depth_raw is not None:arrays['aligned_depth_raw']=aligned_depth_raw
    np.savez_compressed(destination/'measurements.npz',**arrays)
    (destination/'state.json').write_text(encoded)
    return summary


def capture(*,serial=None,bag=None,seconds=2.):
    if bool(serial)==bool(bag):raise ValueError('Choose exactly one explicit device serial or bag recording')
    if not .2<=seconds<=5:raise ValueError('Capture duration must be 0.2–5 seconds')
    try:import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError('尚未安裝 RealSense Python SDK；RGB 預覽可用不代表已具備深度擷取。請依官方 macOS SDK 文件安裝後再執行此工程工具。') from exc
    config=rs.config()
    if bag:
        pipeline=rs.pipeline()
        path=Path(bag).resolve(strict=True)
        config.enable_device_from_file(str(path),repeat_playback=False)
    else:
        if not isinstance(serial,str) or not serial.strip():raise ValueError('An exact serial is required')
        context=rs.context({'inherit':False,'device-mask':int(rs.product_line.D400),'dds':{'enabled':False}})
        pipeline=rs.pipeline(context)
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth,640,480,rs.format.z16,15)
        config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,15)
        # Resolve and verify before starting streams. Reuse this same device:
        # macOS RSUSB can reject a second open of the already claimed interface.
        resolved=config.resolve(rs.pipeline_wrapper(pipeline))
        selected=resolved.get_device()
        if 'D435' not in selected.get_info(rs.camera_info.name) or selected.get_info(rs.camera_info.serial_number)!=serial:
            raise RuntimeError('Selected device is not the explicitly requested D435')
        pipeline.set_device(selected)
    started=False
    try:
        profile=pipeline.start(config);started=True
        device=profile.get_device();name=device.get_info(rs.camera_info.name)
        actual_serial=device.get_info(rs.camera_info.serial_number)
        if 'D435' not in name or (serial and actual_serial!=serial):raise RuntimeError('Selected device is not the explicitly requested D435')
        playback=device.as_playback() if bag else None
        if playback:playback.set_real_time(False)
        frames=select_pair(pipeline,seconds,playback=playback,stopped_status=rs.playback_status.stopped)
        depth=frames.get_depth_frame();color=frames.get_color_frame()
        if not depth or not color:raise RuntimeError('Recording/device must provide native depth and color')
        receipt=time.monotonic();depth_profile=depth.profile.as_video_stream_profile();color_profile=color.profile.as_video_stream_profile()
        def intrinsics(profile):
            p=profile.get_intrinsics()
            return {'width':p.width,'height':p.height,'fx':p.fx,'fy':p.fy,'cx':p.ppx,'cy':p.ppy,
                    'distortion_model':str(p.model),'distortion_coefficients':list(p.coeffs)}
        def timestamp(value):
            return {'sensor_timestamp_ms':value.get_timestamp(),'domain':str(value.get_frame_timestamp_domain()),
                    'frame_number':value.get_frame_number(),'not_exposure_synchronized':True}
        extrinsic=depth_profile.get_extrinsics_to(color_profile)
        scale=depth.get_units()
        cloud=rs.pointcloud().calculate(depth)
        points=np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(-1,3).copy()
        aligned=rs.align(rs.stream.color).process(frames).get_depth_frame()
        color_data=np.asanyarray(color.get_data()).copy()
        if color.profile.format()==rs.format.bgr8:color_data=color_data[:,:,::-1].copy()
        elif color.profile.format()!=rs.format.rgb8:raise RuntimeError('Unsupported color format; RGB8/BGR8 required')
        metadata={'depth_scale_m':scale,'device_name':name,'device_serial':actual_serial,
            'depth_intrinsics':intrinsics(depth_profile),'color_intrinsics':intrinsics(color_profile),
            'aligned_depth_intrinsics':intrinsics(aligned.profile.as_video_stream_profile()),
            'depth_to_color_extrinsics':{'rotation_column_major':list(extrinsic.rotation),'translation_m':list(extrinsic.translation)},
            'depth_timestamp':timestamp(depth),'color_timestamp':timestamp(color),
            'host_receipt_monotonic':receipt,'input_kind':'sdk_bag' if bag else 'explicit_live_d435'}
        return {'depth_raw':np.asanyarray(depth.get_data()).copy(),'color_rgb':color_data,
                'points_m':points,'metadata':metadata,'aligned_depth_raw':np.asanyarray(aligned.get_data()).copy()}
    finally:
        if started:pipeline.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live',action='store_true');mode.add_argument('--bag',type=Path)
    parser.add_argument('--serial');parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.live and not args.serial:parser.error('--live requires --serial; no automatic device selection')
    if args.bag and args.serial:parser.error('--serial is only used with --live')
    destination=args.output or ROOT/'.runtime'/'engineering-observer'/time.strftime('%Y%m%d-%H%M%S')
    try:
        observation=capture(serial=args.serial if args.live else None,bag=args.bag,seconds=args.seconds)
        export_observation(destination,**observation)
    except (ValueError,RuntimeError,OSError) as exc:parser.exit(1,str(exc)+'\n')
    print(destination/'state.json')


if __name__=='__main__':main()
