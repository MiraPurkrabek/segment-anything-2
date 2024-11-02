import subprocess

# List of commands with arguments
commands = [
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.5',
        '--use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--out-filename', '../SAM_ablation/b1',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.5',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--out-filename', '../SAM_ablation/b2',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--out-filename', '../SAM_ablation/b3',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--out-filename', '../SAM_ablation/b4',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'confidence',
        '--num-pos-keypoints', '6',
        '--out-filename', '../SAM_ablation/b5',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--expand-bbox',
        '--out-filename', '../SAM_ablation/b6',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/mask_pose_results/ViTb_pGT.json', '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--conf-thr', '0.8',
        '--no-use-bbox',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '6',
        '--oracle',
        '--out-filename', '../SAM_ablation/b7',
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