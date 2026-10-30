"""Neural reconstruction of recorded drives (milestone 11): a real (or simulated) drive's posed camera images turned
into a scene of 3D Gaussians (Kerbl et al. 2023) that can be rendered from poses the drive never took — the ego
shifted sideways, another camera mount — for "what if" replays of the drive's real appearance. Log replay
(core.logreplay) places CARLA's meshes along a log's trajectories; a reconstruction keeps the log's own pixels, of
the static scene only. docs/reconstruction.md is the manual.

    from falcon_autonomy.core.recon.train import train
    train('kitti_raw:~/kitti/2011_09_26/2011_09_26_drive_0005_sync', '~/demo.falcon/recon/kitti-0005')
    from falcon_autonomy.core.recon.scene import Recon
    rec = Recon.load('~/demo.falcon/recon/kitti-0005')
    out = rec.render(rec.camera(rec.frame(40), lateral_m=1.0))     # image, range, alpha (and semantic)

Modules: raster (the differentiable rasteriser: a pure-PyTorch reference, and raster_triton's kernels for the
compositing on a CUDA GPU), gaussians (parameters, the .ply layout, density
control, Adam), data (posed images from Falcon runs, KITTI raw, nuScenes; masks; initial points), metrics (PSNR,
SSIM, depth), train, scene (a trained reconstruction and its render API), evaluate (held-out frames, other runs,
depth-warped references), rerender (a falcon-raw run from a shifted ego path), vo (monocular visual odometry, bundle
adjustment, intrinsics and the GPS alignment for a dashcam), fromdashcam (a dashcam segment posed, reconstructed and
placed on an imported world), commands (`falcon recon`).

Importing this package imports nothing heavy; torch is imported by the modules that need it (all but data, vo,
fromdashcam and commands), so the GUI imports them only inside the functions that use them.
"""
