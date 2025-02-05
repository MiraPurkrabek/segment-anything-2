import subprocess

# List of commands with arguments
commands = [
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
        '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--GT-is-with-vis',
        '--use-bbox',
        '--no-update-bboxes',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '2',
        '--out-filename', '../BMP_results/SAM_ablation/g19',
        # '--expand-bbox',
        # '--oracle',
        # '--bbox-by-iou', '0.5',
        # '--num-pos-keypoints-if-bbox', '6',
    ],
    [
        'python', 'scripts/batch_pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
        '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--GT-is-with-vis',
        '--no-use-bbox',
        '--no-update-bboxes',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '4',
        '--out-filename', '../BMP_results/SAM_ablation/g20',
        # '--expand-bbox',
        # '--oracle',
        # '--bbox-by-iou', '0.5',
        # '--num-pos-keypoints-if-bbox', '6',
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