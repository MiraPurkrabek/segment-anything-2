import subprocess

# List of commands with arguments
commands = [
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.5',
        '--use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--num-neg-keypoints', '0',
        '--out-filename', '../SAM_ablation/s1',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.5',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--num-neg-keypoints', '0',
        '--out-filename', '../SAM_ablation/s2',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--num-neg-keypoints', '0',
        '--out-filename', '../SAM_ablation/s3',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--num-neg-keypoints', '0',
        '--out-filename', '../SAM_ablation/s4',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--num-neg-keypoints', '1',
        '--out-filename', '../SAM_ablation/s5',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'confidence',
        '--num-pos-keypoints', '6',
        '--num-neg-keypoints', '0',
        '--out-filename', '../SAM_ablation/s6',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--num-neg-keypoints', '0',
        '--expand-bbox',
        '--out-filename', '../SAM_ablation/s7',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--num-neg-keypoints', '0',
        '--oracle',
        '--out-filename', '../SAM_ablation/s8',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--num-neg-keypoints', '0',
        '--crop',
        '--out-filename', '../SAM_ablation/s9',
    ],
]

# Run commands sequentially
for command in commands:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with return code {e.returncode}")
    except Exception as e:
        print(f"An error occurred while running command '{command}': {str(e)}")